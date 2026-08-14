# FEVER PV Forecasting — Interactive Conference Presentation

A cinematic, interactive companion for the physics-aware day-ahead photovoltaic forecasting paper, prepared for the Fourth UK AI Conference 2026.

## Easiest option: open the offline edition

Open this file in Chrome, Edge, Firefox or Safari:

```text
standalone/PV-Forecasting-Conference-Dashboard.html
```

It is a single self-contained file. It embeds its CSS, JavaScript, official logos and generated result figures, so no installation, local server or internet connection is needed during the presentation. Internet is only needed if an external CCAIS, FEVER or portfolio link is clicked.

## Development version

Install Node.js 22.13 or later, then run:

```text
npm install
npm run dev
```

Open the local URL printed in the terminal, normally `http://localhost:3000/`.

Useful checks:

```text
npm run build
npm test
```

## What the presentation contains

- a conference-focused storytelling structure and fullscreen presentation mode;
- interactive missing-data, availability-mask and timestamp-alignment labs;
- random day-fold versus rolling-origin evaluation;
- camera-ready model, representative-week, physics-sensitivity and cost figures;
- verified headline results from the successful private 12-stage run;
- official CCAIS and FEVER visual context;
- explicit single-site, historical-weather and generalisability limitations;
- an offline standalone edition for reliable presentation-room use.

## Verified headline evidence

- random day-fold, all hours: NNLS Stack R² 0.861;
- rolling-origin, all hours: Ridge Stack R² 0.726;
- random day-fold, daylight: R² 0.769;
- rolling-origin, daylight: R² 0.548;
- audited missing hourly PV observations: 9.25%.

## Data safety and scientific scope

This repository contains generated presentation figures but no confidential PV measurements, weather cache, private run directory or source dataset. The underlying study is a single-site FEVER case study using retrospective historical weather. It does not claim external-site validation or operational forecast-vintage performance.

Never add `PV_data.csv`, `weather_cache.csv`, private-data folders or camera-ready run directories to this repository. The ignore rules provide an additional safeguard.

## Credit

Built by [Masood Nazari](https://michaeltheanalyst.github.io/).
