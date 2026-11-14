# VHAGAR scope and portfolio boundary

Updated 17 October 2026.

Portfolio hub: [Lillie US](https://github.com/Ibekwemmanuel7/Lillie_US)

Portfolio map: [Product responsibilities](https://github.com/Ibekwemmanuel7/Lillie_US/blob/main/docs/PORTFOLIO.md)

## Domain and status

**Domain:** Wildfire intelligence

**Status:** Existing research code; Southern validation remains a separate task.

## Current baseline

The existing wildfire research code and its documented detection, burned-area and modelling work remain in this repository. Existing scientific limitations and evaluations continue to apply.

## Planned scope

Southern validation, reliable event delivery, prescribed-burn context and reviewed evidence workflows as defined by the relevant pilot. Portfolio organisation does not establish operational readiness.

## Ownership and interfaces

VHAGAR supplies wildfire observations and burned-area products. Vepris owns forest-management and recovery workflows that may consume them. Vantage owns agricultural decisions.

## Delivery gate

Define the buyer, decision, last useful delivery time, reference data, baseline and acceptance criteria before implementation. Outputs must carry observation and processing dates, provenance, quality, uncertainty and unavailable-data states. Customer-provided records and assessments require explicit access control. No new operational or scientific performance claim follows from adding this roadmap.

The repository owns its reusable domain code, methods, tests and domain documentation. Lillie US owns cross-product integration, shared customer workflows, portfolio configuration and product-release coordination. Do not make this repository depend on Lillie US merely to create a backlink.
