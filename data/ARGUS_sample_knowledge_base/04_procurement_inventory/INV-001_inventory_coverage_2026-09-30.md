---
document_id: INV-001
document_type: inventory_report
as_of_date: 2026-09-30
owner: Supply Planning
confidentiality: internal
status: snapshot
---
# Inventory Coverage Report — 30 September 2026

**Demo data notice:** Fictional inventory values.

| Component | On-hand usable units | Average weekly demand | Estimated coverage |
|---|---:|---:|---:|
| NS-440 power-management module | 18,000 | 3,000 | 6 weeks |
| NS-880 embedded controller | 9,600 | 2,400 | 4 weeks |
| BC-21 sensor interface board | 14,400 | 4,800 | 3 weeks |
| Secure boot controller | 7,500 | 1,500 | 5 weeks |

## Assumptions
Coverage = usable on-hand units divided by average weekly demand. Values exclude unconfirmed inbound orders, quarantined stock, and inventory reserved for other programs.

## Decision caution
Coverage is a planning estimate, not a guaranteed time-to-shortage. Confirm current demand, allocations, yield, and inbound shipment dates before making a supply commitment.
