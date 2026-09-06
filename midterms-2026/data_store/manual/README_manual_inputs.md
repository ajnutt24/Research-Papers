# Hand-maintained inputs

| file | schema | used by |
|---|---|---|
| polls_2026.csv | race_id,pollster,sponsor,partisan,start_date,end_date,sample_size,population,methodology,dem_pct,rep_pct,hypothetical,dem_candidate,rep_candidate | 03 |
| candidates_2026.csv | race_id,party,candidate | 03 (drops polls whose names are not the nominees) |
| rcp_urls.csv | race_id,url | 03 |
| approval_manual.csv | date,approve,disapprove,source | 04 |
| approval_history.csv | seeded by 04; cycle,pres_party,approval,war_salience,weeks_since_escalation,verify | 09 |
| war_salience_2026.csv | date,weeks_since_escalation,attention_index | 04 -> 09 |
| race_ratings_2026.csv | race_id,source,rating | 05 |
| historical_ratings.csv | cycle,race_id,source,rating,dem_won | 05 -> 10, 14 |
| pvi_manual.csv | race_id,pvi (D minus R, Daily Kos, maps in effect Nov 2026) | 06 |
| incumbency_overrides_2026.csv | race_id,status,incumbent_party,notes | 06 |
| fec_totals_manual.csv | race_id,party,individual_contributions | 06 |
| missouri_candidate_map.csv | candidate,party,primary_district,general_district,notes | 06 / README |
| 1976-2022-house.csv, 1976-2020-senate.csv | MIT Election Lab downloads | 07 |
