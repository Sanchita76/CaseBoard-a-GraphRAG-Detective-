# Plain RAG vs GraphRAG

Question: How is Silas Crane connected to the Marlow Museum?

## Plain RAG (top-3 similar sentences)
- [03_felix_ortega.txt] Silas Crane owns Crane Auctions.
- [03_felix_ortega.txt] After the dismissal, Ortega owed a large debt to Silas Crane.
- [05_dockside_warehouse.txt] Silas Crane contacted Nils Arden about renting vans during the winter.

## GraphRAG (follows the connections)
- Connection chain (2 hops): Silas Crane -> Felix Ortega -> Marlow Museum.
- After the dismissal, Ortega owed a large debt to Silas Crane.
- Ortega told friends that he would soon repay Crane.
- Felix Ortega worked as a technician at the Marlow Museum until January.
- Sentinel Security had given Ortega the code while he worked at the museum.
