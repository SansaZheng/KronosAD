# Third-party data attribution and licenses

The repository's MIT code license does not relicense any third-party data.

## Meme Coin research artifact: CC BY 4.0

- Creators: Alberto Maria Mongardini and Alessandro Mei.
- Title: *A Midsummer Meme's Dream: Investigating Market Manipulations in the Meme Coin Ecosystem*.
- Venue: USENIX Security 2026.
- Version: Zenodo v1, published 2025-12-05, DOI https://doi.org/10.5281/zenodo.17830944 .
- Original artifact: https://zenodo.org/api/records/17830944/files/meme_coin-main.zip/content .
- Repository: https://github.com/am-mong/Code-and-Dataset-for-A-Midsummer-Meme-s-Dream- .
- License: Creative Commons Attribution 4.0 International,
  https://creativecommons.org/licenses/by/4.0/ .
- Publisher license metadata is preserved in `raw/meme2026/zenodo_record.json`.
- Modifications: column names canonicalized; time ascending and UTC; split at missing intervals;
  retained contiguous runs of at least 600 rows; quote currency units rescaled from the first
  64 observed bars; source quote-volume interpreted as amount under API default USD convention;
  author pump_start/dump_end intervals joined into clearly identified algorithmic labels.
- No creator endorsement of this adaptation is implied. Original archive remains unchanged.

Suggested citation:

```bibtex
@misc{mongardini2025midsummerartifact,
  author = {Mongardini, Alberto Maria and Mei, Alessandro},
  title = {Code and Dataset for A Midsummer Meme's Dream:
           Investigating Market Manipulations in the Meme Coin Ecosystem},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.17830944}
}
```

## Sapienza event list: MIT

- Authors: Massimo La Morgia, Alessandro Mei, Francesco Sassi, Julinda Stefa.
- Title: *Pump and Dumps in the Bitcoin Era: Real Time Detection of Cryptocurrency Market Manipulations*.
- Venue: ICCCN 2020, pages 1–9, DOI https://doi.org/10.1109/ICCCN49398.2020.9209660 .
- Source: https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset .
- Pinned commit: `d71250d4cb055dde2d415c8cba38a0dcd6eb6e16`.
- Original MIT license and copyright notice are retained in `raw/sapienza/LICENSE`.
- Modifications: Binance events deduplicated and 12 starter cases selected deterministically;
  announcement times joined to independent exchange OHLCVA with explicit weak interval labels.

## Binance Vision market data: CC BY-NC-SA 4.0 + dataset terms

- Data provider: Binance Vision / Binance Public Data.
- Collection: https://data.binance.vision/ .
- Source documentation: https://github.com/binance/binance-public-data .
- Terms snapshot: `raw/binance/TERMS_AND_CONDITIONS.md`, version 1.0, updated 2026-08-26.
- License stated in those terms: Creative Commons Attribution-NonCommercial-ShareAlike 4.0,
  https://creativecommons.org/licenses/by-nc-sa/4.0/ , plus the provider's dataset terms.
- This adaptation and redistribution is for non-commercial research and reproducibility.
  The derived Binance OHLCVA and joined event-window data retain CC BY-NC-SA 4.0.
  Do not treat them as MIT, commercially licensed data, or a live trading product.
- Modifications: ZIP members parsed; milliseconds/microseconds interpreted according to archive era;
  quote asset volume mapped to amount; sorted UTC bars; separated at real time gaps;
  BTC-quoted price/amount converted to satoshi; source event-window labels added where stated;
  recent unknown labels explicitly set to -1.
- Every daily ZIP is accompanied by its publisher's SHA256 `.CHECKSUM`.
- Complete per-file source URLs and local hashes: `raw_manifest.json`.
- No affiliation with or endorsement by Binance is implied.
