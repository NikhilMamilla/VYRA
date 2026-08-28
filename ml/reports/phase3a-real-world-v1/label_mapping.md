# VizWiz -> VYRA label mapping

Categories: **A** directly mappable · **B** partially · **C** not mappable · **D** auxiliary only.

| VizWiz | meaning | VYRA label | category | confidence | reasoning |
|---|---|---|---|---|---|
| BLR | blurry / out of focus / motion blur | blur | A | high | VizWiz BLR (blurry / out of focus / motion) is the same concept as VYRA 'blur'. Both defocus and motion blur are covered on each side. |
| BRT | too bright / overexposed | overexposure | A | medium | 'Too bright' corresponds to VYRA 'overexposure'. Slight gap: BRT can also be flagged for glare/reflections that are not global overexposure. |
| DRK | too dark / underexposed | underexposure | A | medium | 'Too dark' corresponds to VYRA 'underexposure'. Real dark images also carry heavy sensor noise, which VYRA models as a separate label. |
| OBS | obstruction - finger or object partly covering the lens | defect | B | low | OBS (finger/object over the lens) overlaps ONLY the 'occlusion' sub-type of VYRA's synthetic 'defect' (1 of 5 synthetic defect types). VYRA defect also covers dead pixels, banding, block corruption, colour blotches -- none of which VizWiz labels. Treated as a weak proxy, reported with heavy caveats. |
| FRM | framing - subject cut off / incomplete / bad composition | — | C | n/a | Framing / subject cut off. VYRA has no framing concept and it is not an image-quality degradation in VYRA's taxonomy. Unmapped. |
| ROT | wrong rotation / orientation | — | C | n/a | Wrong orientation. Not a VYRA concept and not a degradation. Unmapped. |
| OTH | other quality issue (free-text) | — | D | n/a | Free-text 'other'. Too heterogeneous to map; kept only as context. |
| NON | no quality issue | — | D | n/a | 'No issue' vote count. Auxiliary: high NON is corroborating evidence for a clean image (used in analysis, not as a label). |
| — | — | noise | C | n/a | VizWiz has no noise category. Real sensor noise is present in many dark images but is never annotated. Not directly supported by this dataset. |
| — | — | corruption | C | n/a | VizWiz has no compression / digital-corruption category. 'unrecognizable' is severity, not corruption (see note). Not directly supported by this dataset. |

**Evaluable VYRA labels:** blur, overexposure, underexposure, defect.

**Not supported by VizWiz** (never scored, no fake negatives): `noise`, `corruption`.
