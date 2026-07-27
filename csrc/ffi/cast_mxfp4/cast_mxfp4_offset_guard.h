// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Pure host arithmetic for selecting the guarded MXFP4 uint32 offset kernel.
// This header intentionally has no HIP/XLA dependencies so its full boundary
// contract can be exercised by a normal C++ unit test without a GPU.

#pragma once

#include <climits>
#include <cstdint>
#include <limits>

namespace mxfp4::offset_guard {

enum class GuardStatus {
  kOk,
  kNonPositiveDimension,
  kDimensionExceedsInt,
  kDimensionMisaligned,
  kDerivedValueExceedsInt,
  kArithmeticOverflow,
};

enum class OffsetMode : int {
  kOff = 0,
  kAuto = 1,
  kForce64 = 2,
};

enum class OffsetType {
  kU64,
  kU32,
};

struct LayoutFlags {
  bool use_rowwise;
  bool use_colwise;
  bool shuffle_scales;
  bool shuffle_rowwise_fp4;
  bool shuffle_colwise_fp4;
};

struct TemplateFlags {
  bool use_rowwise;
  bool use_colwise;
  bool shuffle_scales;
  bool use_hadamard_row;
  bool use_hadamard_col;
  bool shuffle_rowwise_fp4;
  bool shuffle_colwise_fp4;
  bool use_sr_row;
  bool use_sr_col;
};

struct GuardResult {
  GuardStatus status = GuardStatus::kOk;
  bool u32_safe = false;
  uint64_t input_last_byte = 0;
  uint64_t row_fp4_last_byte = 0;
  uint64_t col_fp4_last_byte = 0;
  uint64_t row_scale_last_byte = 0;
  uint64_t col_scale_last_byte = 0;
  uint64_t m_pad = 0;
  uint64_t k_pad = 0;
  uint64_t row_scale_n = 0;
  uint64_t row_scale_n_pad = 0;
  uint64_t col_scale_n = 0;
  uint64_t col_scale_n_pad = 0;
};

struct OffsetSelection {
  bool valid_mode;
  OffsetType type;
};

inline bool checked_add_u64(uint64_t lhs, uint64_t rhs, uint64_t* out) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
    return false;
  }
  *out = lhs + rhs;
  return true;
}

inline bool checked_mul_u64(uint64_t lhs, uint64_t rhs, uint64_t* out) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    return false;
  }
  *out = lhs * rhs;
  return true;
}

inline bool checked_shift_left_u64(
    uint64_t value, unsigned shift, uint64_t* out) {
  if (shift >= 64 ||
      value > (std::numeric_limits<uint64_t>::max() >> shift)) {
    return false;
  }
  *out = value << shift;
  return true;
}

inline bool checked_span_last_byte(
    uint64_t first, uint64_t second, uint64_t element_bytes,
    uint64_t* last_byte) {
  uint64_t elements = 0;
  uint64_t bytes = 0;
  if (!checked_mul_u64(first, second, &elements) ||
      !checked_mul_u64(elements, element_bytes, &bytes) || bytes == 0) {
    return false;
  }
  *last_byte = bytes - 1;
  return true;
}

inline bool fits_uint32_endpoint(uint64_t last_byte) {
  return last_byte <= std::numeric_limits<uint32_t>::max();
}

inline bool check_fp4_shuffle_helper(
    uint64_t rows, uint64_t packed_width, uint64_t* max_index) {
  if (rows == 0 || packed_width < 2 ||
      rows > static_cast<uint64_t>(INT_MAX) ||
      packed_width > static_cast<uint64_t>(INT_MAX)) {
    return false;
  }

  const uint64_t row = rows - 1;
  const uint64_t col = packed_width - 2;
  const uint64_t n_block = row >> 4;
  const uint64_t row_in_block = row & 15;
  const uint64_t k_block = col >> 5;
  const uint64_t col_in_block = col & 31;
  const uint64_t sub_block = col_in_block >> 4;
  const uint64_t k_elem = col_in_block & 15;
  const uint64_t limit = static_cast<uint64_t>(INT_MAX);

  uint64_t packed_stride = 0;
  uint64_t term = 0;
  uint64_t total = 0;
  if (!checked_shift_left_u64(packed_width, 4, &packed_stride) ||
      packed_stride > limit ||
      !checked_mul_u64(n_block, packed_stride, &total) ||
      total > limit ||
      !checked_mul_u64(k_block, 512, &term) || term > limit ||
      !checked_add_u64(total, term, &total) || total > limit ||
      !checked_mul_u64(sub_block, 256, &term) || term > limit ||
      !checked_add_u64(total, term, &total) || total > limit ||
      !checked_mul_u64(row_in_block, 16, &term) || term > limit ||
      !checked_add_u64(total, term, &total) || total > limit ||
      !checked_add_u64(total, k_elem, &total) || total > limit) {
    return false;
  }
  *max_index = total;
  return true;
}

inline bool check_scale_shuffle_helper(
    uint64_t rows, uint64_t logical_cols, uint64_t padded_cols,
    uint64_t* max_index) {
  if (rows == 0 || logical_cols == 0 || logical_cols > padded_cols ||
      padded_cols < 8 || (padded_cols & 7) != 0 ||
      rows > static_cast<uint64_t>(INT_MAX) ||
      logical_cols > static_cast<uint64_t>(INT_MAX) ||
      padded_cols > static_cast<uint64_t>(INT_MAX)) {
    return false;
  }

  // The row and column permutations are separable, but neither is monotonic
  // inside its 32-row / 8-column group. The final group has the greatest
  // base (each group step is at least 256, above either local maximum), so
  // find the true local maximum over that possibly partial final group.
  const uint64_t last_row = rows - 1;
  const uint64_t last_col = logical_cols - 1;
  const uint64_t row_group = last_row >> 5;
  const uint64_t col_group = last_col >> 3;
  const uint64_t rows_in_last_group = (last_row & 31) + 1;
  const uint64_t cols_in_last_group = (last_col & 7) + 1;
  const uint64_t scale_groups = padded_cols >> 3;
  const uint64_t limit = static_cast<uint64_t>(INT_MAX);

  uint64_t max_row_local = 0;
  for (uint64_t row = 0; row < rows_in_last_group; ++row) {
    uint64_t local = 0;
    if (!checked_shift_left_u64(row & 15, 2, &local) ||
        !checked_add_u64(local, (row >> 4) & 1, &local)) {
      return false;
    }
    if (local > max_row_local) {
      max_row_local = local;
    }
  }

  uint64_t max_col_local = 0;
  for (uint64_t col = 0; col < cols_in_last_group; ++col) {
    uint64_t local = 0;
    uint64_t term = 0;
    if (!checked_shift_left_u64(col & 3, 6, &local) ||
        !checked_shift_left_u64((col >> 2) & 1, 1, &term) ||
        !checked_add_u64(local, term, &local)) {
      return false;
    }
    if (local > max_col_local) {
      max_col_local = local;
    }
  }

  uint64_t product = 0;
  uint64_t term = 0;
  uint64_t total = 0;
  if (!checked_mul_u64(row_group, scale_groups, &product) ||
      product > limit ||
      !checked_shift_left_u64(product, 8, &total) || total > limit ||
      !checked_shift_left_u64(col_group, 8, &term) || term > limit ||
      !checked_add_u64(total, term, &total) || total > limit ||
      !checked_add_u64(total, max_row_local, &total) || total > limit ||
      !checked_add_u64(total, max_col_local, &total) || total > limit) {
    return false;
  }
  *max_index = total;
  return true;
}

inline GuardResult evaluate_offset_guard(
    int64_t m_wide, int64_t k_wide, const LayoutFlags& flags) {
  GuardResult result;
  if (m_wide <= 0 || k_wide <= 0) {
    result.status = GuardStatus::kNonPositiveDimension;
    return result;
  }
  if (m_wide > INT_MAX || k_wide > INT_MAX) {
    result.status = GuardStatus::kDimensionExceedsInt;
    return result;
  }
  if ((m_wide % 32) != 0 || (k_wide % 32) != 0) {
    result.status = GuardStatus::kDimensionMisaligned;
    return result;
  }

  const uint64_t m = static_cast<uint64_t>(m_wide);
  const uint64_t k = static_cast<uint64_t>(k_wide);
  const uint64_t m_pad_units = m / 256 + (m % 256 != 0);
  const uint64_t k_pad_units = k / 256 + (k % 256 != 0);
  result.row_scale_n = k / 32;
  result.col_scale_n = m / 32;
  const uint64_t row_pad_units =
      result.row_scale_n / 8 + (result.row_scale_n % 8 != 0);
  const uint64_t col_pad_units =
      result.col_scale_n / 8 + (result.col_scale_n % 8 != 0);

  if (!checked_mul_u64(m_pad_units, 256, &result.m_pad) ||
      !checked_mul_u64(k_pad_units, 256, &result.k_pad) ||
      !checked_mul_u64(row_pad_units, 8, &result.row_scale_n_pad) ||
      !checked_mul_u64(col_pad_units, 8, &result.col_scale_n_pad)) {
    result.status = GuardStatus::kArithmeticOverflow;
    return result;
  }
  const uint64_t int_limit = static_cast<uint64_t>(INT_MAX);
  if (result.m_pad > int_limit || result.k_pad > int_limit ||
      result.row_scale_n > int_limit ||
      result.row_scale_n_pad > int_limit ||
      result.col_scale_n > int_limit ||
      result.col_scale_n_pad > int_limit) {
    result.status = GuardStatus::kDerivedValueExceedsInt;
    return result;
  }

  if (!checked_span_last_byte(m, k, 2, &result.input_last_byte) ||
      (flags.use_rowwise &&
       (!checked_span_last_byte(
           m, k / 2, 1, &result.row_fp4_last_byte) ||
        !checked_span_last_byte(
            result.m_pad, result.row_scale_n_pad, 1,
            &result.row_scale_last_byte))) ||
      (flags.use_colwise &&
       (!checked_span_last_byte(
           k, m / 2, 1, &result.col_fp4_last_byte) ||
        !checked_span_last_byte(
            result.k_pad, result.col_scale_n_pad, 1,
            &result.col_scale_last_byte)))) {
    result.status = GuardStatus::kArithmeticOverflow;
    return result;
  }

  result.u32_safe = fits_uint32_endpoint(result.input_last_byte);
  if (flags.use_rowwise) {
    result.u32_safe =
        result.u32_safe &&
        fits_uint32_endpoint(result.row_fp4_last_byte) &&
        fits_uint32_endpoint(result.row_scale_last_byte);
  }
  if (flags.use_colwise) {
    result.u32_safe =
        result.u32_safe &&
        fits_uint32_endpoint(result.col_fp4_last_byte) &&
        fits_uint32_endpoint(result.col_scale_last_byte);
  }

  uint64_t helper_max = 0;
  if (flags.use_rowwise && flags.shuffle_rowwise_fp4) {
    result.u32_safe =
        result.u32_safe &&
        check_fp4_shuffle_helper(m, k / 2, &helper_max);
  }
  if (flags.use_colwise && flags.shuffle_colwise_fp4) {
    result.u32_safe =
        result.u32_safe &&
        check_fp4_shuffle_helper(k, m / 2, &helper_max);
  }
  if (flags.shuffle_scales && flags.use_rowwise) {
    result.u32_safe =
        result.u32_safe &&
        check_scale_shuffle_helper(
            m, result.row_scale_n, result.row_scale_n_pad, &helper_max);
  }
  if (flags.shuffle_scales && flags.use_colwise) {
    result.u32_safe =
        result.u32_safe &&
        check_scale_shuffle_helper(
            k, result.col_scale_n, result.col_scale_n_pad, &helper_max);
  }
  return result;
}

inline bool is_u32_specialized_template(const TemplateFlags& flags) {
  return
      flags.use_rowwise && flags.use_colwise && flags.shuffle_scales &&
      flags.use_hadamard_row && flags.use_hadamard_col &&
      !flags.shuffle_rowwise_fp4 && !flags.use_sr_row && !flags.use_sr_col;
}

inline OffsetSelection select_offset_type(
    int requested_mode, bool u32_safe, bool specialized_template) {
  if (requested_mode == static_cast<int>(OffsetMode::kOff) ||
      requested_mode == static_cast<int>(OffsetMode::kForce64)) {
    return {true, OffsetType::kU64};
  }
  if (requested_mode == static_cast<int>(OffsetMode::kAuto)) {
    return {
        true,
        u32_safe && specialized_template ? OffsetType::kU32
                                         : OffsetType::kU64,
    };
  }
  return {false, OffsetType::kU64};
}

}  // namespace mxfp4::offset_guard
