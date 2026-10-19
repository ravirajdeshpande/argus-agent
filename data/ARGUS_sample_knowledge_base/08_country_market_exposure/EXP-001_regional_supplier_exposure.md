---
document_id: EXP-001
document_type: country_market_exposure
owner: Global Supply Strategy
as_of_date: 2026-09-18
confidentiality: internal
status: snapshot
---
# Regional Supplier and Manufacturing Exposure

**Demo data notice:** Fictional exposure map.

| Product | Component | Supplier | Manufacturing region | Business dependency |
|---|---|---|---|---|
| Orion Edge Gateway | NS-440 | SUP-100 | Taiwan | Single-source power module |
| Orion Edge Gateway | Secure boot controller | SUP-130 | South Korea | Conditional supplier approval |
| Atlas Vision Controller | NS-880 | SUP-100 | Taiwan; Singapore | Controller dependency |
| Atlas Vision Controller | BC-21 | SUP-110 | Malaysia; Vietnam (secondary site) | Site-specific approval required |

## Interpretation
Manufacturing location, shipping origin, supplier headquarters, and customer destination are different attributes. An event affecting one country should be mapped against the relevant transaction and actual production site.

## Recommended checks
Verify current site, origin documentation, route, supplier status, product dependency, and applicable regulation before concluding that a product is exposed.
