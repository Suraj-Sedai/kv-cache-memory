# Commit map: pre-rewrite hashes → published hashes

On 2026-10-17 the history was rewritten with git filter-repo to remove commit trailers, which
changed the hash of every commit from 2026-06-15 onward (68 commits) while leaving each commit's
tree unchanged; the 16 earlier commits kept their hashes.

Results recorded before the rewrite name the old hashes. Every old hash cited in this repository
maps to a published commit with an identical tree:

| Old hash | Published | Matched by |
|---|---|---|
| e383030 (also c82b120) | 017b63a | filter-repo commit map; identical tree and message |
| 0d029d6 | 40f6b89 | filter-repo commit map; identical tree and message |
| 237ad2d (also 98d0e75) | d0016a1 | filter-repo commit map; identical tree and message |
| fceab20 (also 90ed952) | d1207bf | filter-repo commit map; identical tree and message |
| e414dee | 0e845af | filter-repo commit map; identical tree and message |
| ae0cb58 | d4f4678 | filter-repo commit map; identical tree and message |
| fb29cd5 | 7c3c34c | filter-repo commit map; identical tree and message |
| feb71b7 | f8aa452 | local-only commit, never published; tree compared byte for byte before it was deleted |
| 8251fad | 4b56d3e | local-only commit, never published; tree compared byte for byte before it was deleted |
| 00a2401 | 6343c99 | local-only commit, never published; tree compared byte for byte before it was deleted |

A `-dirty` suffix means the run started with uncommitted changes on top of that commit; the
suffix is kept on the published hash.

Hashes unchanged by the rewrite and cited as-is: 0584aa1, 13fbfae, 2a86eeb, eae781c, f7826fa.

## Raw records edited in place

Commits 001a31b and 0a5d1a6 (2026-10-17) replaced the old hashes with the published ones inside
these raw records. No other field changed; the old value in each is the left column above.

| File | Field | Was | Now |
|---|---|---|---|
| results/activation_census/activation_census.json.env.json | git_commit | 00a2401-dirty | 6343c99-dirty |
| results/matmul_peak/matmul_peak.json.env.json | git_commit | feb71b7 | f8aa452 |
| results/paired_layout_check/paired_layout_check.json.env.json | git_commit | feb71b7 | f8aa452 |
| results/sdpa_backend_check/sdpa_backend_check.json.env.json | git_commit | feb71b7 | f8aa452 |
| results/block_size_sweep/noise_check.json | commit | 8251fad | 4b56d3e |
| results/block_size_sweep/noise_check.json | key of the pre-fix throughput | e383030 | 017b63a |
| results/pre_fix/wddm_spill.json | meta.commit | 8251fad-dirty | 4b56d3e-dirty |
| results/cache_comparison/exp1_layout_comparison.csv | commit column, all 40 rows | fb29cd5-dirty | 7c3c34c-dirty |
| results/pre_fix/exp{1,2,3}_*_e383030.csv | file name | _e383030 | _017b63a |

The values before the edit are in the parent of each commit (`git show 001a31b^:<path>`,
`git show 0a5d1a6^:<path>`).
