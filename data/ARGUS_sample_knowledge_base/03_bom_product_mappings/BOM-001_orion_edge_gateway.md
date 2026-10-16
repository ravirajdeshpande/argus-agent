---
document_id: BOM-001
document_type: bill_of_materials
product: Orion Edge Gateway
revision: C
effective_date: 2026-07-01
confidentiality: internal
status: current-demo-revision
---
# Bill of Materials — Orion Edge Gateway, Revision C

**Demo data notice:** Fictional product and component data.

| Component ID | Component | Supplier ID | Region | Units per product | Substitution status |
|---|---|---|---|---:|---|
| CMP-440 | NS-440 power-management module | SUP-100 | Taiwan | 1 | Engineering review required |
| CMP-441 | Secure boot controller | SUP-130 | South Korea | 1 | Conditional source |
| CMP-442 | Aluminum enclosure | SUP-120 | India | 1 | Approved alternate exists |
| CMP-443 | Ethernet interface | SUP-140 | Japan | 2 | No alternate recorded |

## Dependency notes
- CMP-440 is a single-source critical component in this revision.
- A change to CMP-441 requires security validation and supplier approval.
- The BOM does not include on-hand inventory; use the latest inventory report for coverage calculations.

## ARGUS analysis
A restriction affecting the NS-440 component could affect Orion production if the rule applies to the actual component, transaction, destination, and parties. Verify scope before assigning exposure.
