-- Frozen eligible_v1 rules. Reproduction only: do not overwrite the existing table.
-- Preserve the source fire-grouped split: 2015-2017 train, 2018-2019 validation,
-- 2020 test. No split reassignment, deduplication, or null imputation.
CREATE TABLE `ee-wildfire-recovery-vasant.phase3_ca175_tile1_v1.patch_model_eligible_v1` AS
SELECT * REPLACE ('eligible_v1' AS eligibility_status)
FROM `ee-wildfire-recovery-vasant.phase3_ca175_tile1_v1.patch_candidates_v1`
WHERE forest_fraction >= 0.50
  AND fire_overlap_fraction >= 0.50
  AND mtbs_meaningful_burn_fraction >= 0.25
  AND nbr_pre_valid_fraction >= 0.80
  AND nbr_1_valid_fraction >= 0.80
  AND nbr_5_valid_fraction >= 0.80
  AND prefire_selected_year_count_mean >= 1
  AND season1_obs_count_mean >= 1
  AND season5_obs_count_mean >= 1
  AND prior_fire_fraction_10yr = 0
  AND reburn_fraction = 0
  AND lcms_pre_10yr_clear_disturbance_fraction < 0.05
  AND lcms_post_y_to_y5_clear_disturbance_fraction < 0.05
  AND lcms_pre_10yr_complete_coverage_fraction >= 0.80
  AND lcms_post_y_to_y5_complete_coverage_fraction >= 0.80;
