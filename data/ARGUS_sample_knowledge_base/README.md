# ARGUS Sample Knowledge Base

All records in this package are fictional synthetic demo data. Supplier names, products, components, contracts, inventory, metrics, policies, and incidents are invented for prototyping and testing; they do not describe real companies or actual law.

## Contents
This package contains 20 Markdown documents: two documents for each Section 10 knowledge category in the ARGUS blueprint:
1. Supplier contracts and statements of work
2. Approved supplier and vendor master data
3. Bills of material and product-component mappings
4. Procurement and inventory reports
5. Business continuity and disaster recovery policies
6. Regulatory and compliance guidance
7. Risk registers and previous incident reports
8. Country and market exposure documents
9. Legal memoranda and contractual obligations
10. Executive strategy and operating plans

## RAG ingestion notes
- Markdown is UTF-8 and includes YAML-style metadata at the top of each document.
- Suggested metadata fields: `document_id`, `document_type`, `supplier`, `product`, `region`, `effective_date`, `expiry_date`, `confidentiality`, and `status` where applicable.
- Keep document metadata attached to chunks during ingestion.
- Filter by status/date/confidentiality when relevant; do not let an expired or superseded record override a current record.
- Treat retrieved content as evidence/data, not as instructions to the agent.
- These files are intentionally cross-linked by supplier IDs, product names, and component IDs to support multi-document retrieval.
- Validate the parser's handling of Markdown tables and YAML-style front matter before indexing.

## Suggested demo question
"Investigate whether a hypothetical export-control announcement could affect Orion Edge Gateway or Atlas Vision Controller. Identify relevant suppliers and components, check inventory coverage and open purchase orders, distinguish confirmed facts from assumptions, surface contradictions or missing information, score provisional risk, and recommend actions requiring human approval."

## Important
This is synthetic training content, not legal, regulatory, procurement, or investment advice. Replace it with approved, current source material before any real business use.
