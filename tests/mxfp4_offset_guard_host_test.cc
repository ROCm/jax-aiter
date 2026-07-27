// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <cassert>
#include <climits>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>

#include "cast_mxfp4_offset_guard.h"

namespace guard = mxfp4::offset_guard;

static void test_checked_arithmetic() {
  uint64_t out = 0;
  assert(guard::checked_add_u64(7, 9, &out) && out == 16);
  assert(!guard::checked_add_u64(std::numeric_limits<uint64_t>::max(), 1,
                                 &out));
  assert(guard::checked_mul_u64(7, 9, &out) && out == 63);
  assert(!guard::checked_mul_u64(std::numeric_limits<uint64_t>::max(), 2,
                                 &out));
  assert(guard::checked_shift_left_u64(3, 4, &out) && out == 48);
  assert(!guard::checked_shift_left_u64(
      std::numeric_limits<uint64_t>::max(), 1, &out));
}

static void test_span_endpoint_boundaries() {
  uint64_t last = 0;
  assert(guard::checked_span_last_byte(32768, 65536, 2, &last));
  assert(last == std::numeric_limits<uint32_t>::max());
  assert(guard::fits_uint32_endpoint(last));

  assert(guard::checked_span_last_byte(32768, 65568, 2, &last));
  assert(last > std::numeric_limits<uint32_t>::max());
  assert(!guard::fits_uint32_endpoint(last));
  assert(!guard::checked_span_last_byte(
      std::numeric_limits<uint64_t>::max(), 2, 1, &last));
}

static void test_shuffle_helper_boundaries() {
  uint64_t max_index = 0;

  // Scale shuffle is non-monotonic inside each 8-column group:
  // columns 0..4 map to local offsets 0,64,128,192,2. Therefore the true
  // maximum for five logical columns is at row 31, column 3, not at the
  // final logical column.
  assert(guard::check_scale_shuffle_helper(32, 5, 8, &max_index));
  assert(max_index == 253);

  // The row permutation is also non-monotonic at the 16-row boundary.
  assert(guard::check_scale_shuffle_helper(17, 5, 8, &max_index));
  assert(max_index == 252);
  assert(guard::check_scale_shuffle_helper(33, 5, 8, &max_index));
  assert(max_index == 448);

  // Crossing an 8-column group must account for the new 256-byte group base.
  assert(guard::check_scale_shuffle_helper(32, 9, 16, &max_index));
  assert(max_index == 317);

  // FP4 helper: 32768 * 65536 bytes has a final two-byte store beginning at
  // INT_MAX-1. The next 32-byte packed stride crosses the signed-int contract.
  assert(guard::check_fp4_shuffle_helper(32768, 65536, &max_index));
  assert(max_index == static_cast<uint64_t>(INT_MAX) - 1);
  assert(!guard::check_fp4_shuffle_helper(32768, 65568, &max_index));

  // Scale helper: a full 32768x65536 shuffled domain ends exactly at INT_MAX.
  assert(guard::check_scale_shuffle_helper(
      32768, 65536, 65536, &max_index));
  assert(max_index == static_cast<uint64_t>(INT_MAX));
  assert(guard::check_scale_shuffle_helper(
      32768, 65533, 65536, &max_index));
  assert(max_index == static_cast<uint64_t>(INT_MAX) - 2);
  assert(!guard::check_scale_shuffle_helper(
      32768, 65544, 65544, &max_index));
  assert(!guard::check_scale_shuffle_helper(32, 9, 8, &max_index));
  assert(!guard::check_scale_shuffle_helper(32, 5, 7, &max_index));
  assert(!guard::check_scale_shuffle_helper(32, 5, 9, &max_index));
}

static uint64_t scale_shuffle_index(
    uint64_t row, uint64_t col, uint64_t padded_cols) {
  return (((row >> 5) * (padded_cols >> 3)) << 8) +
         ((col >> 3) << 8) + ((col & 3) << 6) +
         ((row & 15) << 2) + (((col >> 2) & 1) << 1) +
         ((row >> 4) & 1);
}

static void test_scale_shuffle_true_max_exhaustive_small_domains() {
  for (uint64_t padded_cols = 8; padded_cols <= 64; padded_cols += 8) {
    for (uint64_t rows = 1; rows <= 65; ++rows) {
      for (uint64_t logical_cols = 1; logical_cols <= padded_cols;
           ++logical_cols) {
        uint64_t expected = 0;
        for (uint64_t row = 0; row < rows; ++row) {
          for (uint64_t col = 0; col < logical_cols; ++col) {
            const uint64_t index =
                scale_shuffle_index(row, col, padded_cols);
            if (index > expected) {
              expected = index;
            }
          }
        }
        uint64_t actual = 0;
        assert(guard::check_scale_shuffle_helper(
            rows, logical_cols, padded_cols, &actual));
        assert(actual == expected);
      }
    }
  }
}

static void test_production_shapes_and_pointer_endpoint() {
  constexpr guard::LayoutFlags activation{
      true, true, true, false, true};
  constexpr guard::LayoutFlags gradient{
      true, true, true, false, false};

  for (int64_t k : {int64_t{4096}, int64_t{14336}}) {
    const auto result = guard::evaluate_offset_guard(32768, k, activation);
    assert(result.status == guard::GuardStatus::kOk);
    assert(result.u32_safe);
  }
  for (int64_t k : {int64_t{1024}, int64_t{4096}, int64_t{14336}}) {
    const auto result = guard::evaluate_offset_guard(32768, k, gradient);
    assert(result.status == guard::GuardStatus::kOk);
    assert(result.u32_safe);
  }

  const auto exact =
      guard::evaluate_offset_guard(32768, 65536, activation);
  assert(exact.status == guard::GuardStatus::kOk);
  assert(exact.input_last_byte == std::numeric_limits<uint32_t>::max());
  assert(exact.u32_safe);

  const auto above =
      guard::evaluate_offset_guard(32768, 65568, activation);
  assert(above.status == guard::GuardStatus::kOk);
  assert(!above.u32_safe);
}

static void test_contract_errors_without_allocation() {
  constexpr guard::LayoutFlags gradient{
      true, true, true, false, false};

  assert(guard::evaluate_offset_guard(0, 4096, gradient).status ==
         guard::GuardStatus::kNonPositiveDimension);
  assert(guard::evaluate_offset_guard(-32, 4096, gradient).status ==
         guard::GuardStatus::kNonPositiveDimension);
  assert(guard::evaluate_offset_guard(
             static_cast<int64_t>(INT_MAX) + 1, 4096, gradient)
             .status == guard::GuardStatus::kDimensionExceedsInt);
  assert(guard::evaluate_offset_guard(32, 33, gradient).status ==
         guard::GuardStatus::kDimensionMisaligned);

  // Largest positive multiple of 32 below INT_MAX pads to INT_MAX+1 at 256.
  assert(guard::evaluate_offset_guard(
             static_cast<int64_t>(INT_MAX) - 31, 32, gradient)
             .status == guard::GuardStatus::kDerivedValueExceedsInt);
}

static void test_template_eligibility_and_mode_selection() {
  constexpr guard::TemplateFlags activation{
      true, true, true, true, true, false, true, false, false};
  constexpr guard::TemplateFlags gradient{
      true, true, true, true, true, false, false, false, false};
  constexpr guard::TemplateFlags weight{
      true, true, true, false, false, true, true, false, false};
  constexpr guard::TemplateFlags activation_sr{
      true, true, true, true, true, false, true, true, false};

  assert(guard::is_u32_specialized_template(activation));
  assert(guard::is_u32_specialized_template(gradient));
  assert(!guard::is_u32_specialized_template(weight));
  assert(!guard::is_u32_specialized_template(activation_sr));

  auto selected = guard::select_offset_type(0, true, true);
  assert(selected.valid_mode && selected.type == guard::OffsetType::kU64);
  selected = guard::select_offset_type(2, true, true);
  assert(selected.valid_mode && selected.type == guard::OffsetType::kU64);
  selected = guard::select_offset_type(1, true, true);
  assert(selected.valid_mode && selected.type == guard::OffsetType::kU32);
  selected = guard::select_offset_type(1, false, true);
  assert(selected.valid_mode && selected.type == guard::OffsetType::kU64);
  selected = guard::select_offset_type(1, true, false);
  assert(selected.valid_mode && selected.type == guard::OffsetType::kU64);
  selected = guard::select_offset_type(99, true, true);
  assert(!selected.valid_mode);
}

int main() {
  test_checked_arithmetic();
  test_span_endpoint_boundaries();
  test_shuffle_helper_boundaries();
  test_scale_shuffle_true_max_exhaustive_small_domains();
  test_production_shapes_and_pointer_endpoint();
  test_contract_errors_without_allocation();
  test_template_eligibility_and_mode_selection();
  constexpr guard::LayoutFlags activation{
      true, true, true, false, true};
  const auto exact =
      guard::evaluate_offset_guard(32768, 65536, activation);
  const auto oversized =
      guard::evaluate_offset_guard(32768, 65568, activation);
  const auto exact_selection =
      guard::select_offset_type(1, exact.u32_safe, true);
  const auto oversized_selection =
      guard::select_offset_type(1, oversized.u32_safe, true);
  const auto forced_selection =
      guard::select_offset_type(2, exact.u32_safe, true);
  std::cout
      << "{\"status\":\"PASS\",\"exact_endpoint\":"
      << exact.input_last_byte
      << ",\"exact_auto_u32\":"
      << (exact_selection.type == guard::OffsetType::kU32 ? "true" : "false")
      << ",\"oversized_endpoint\":"
      << oversized.input_last_byte
      << ",\"oversized_auto_fallback_u64\":"
      << (oversized_selection.type == guard::OffsetType::kU64 ? "true"
                                                               : "false")
      << ",\"force64_u64\":"
      << (forced_selection.type == guard::OffsetType::kU64 ? "true" : "false")
      << "}\n";
  return 0;
}
