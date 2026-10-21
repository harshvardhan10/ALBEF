# Cleanup inventory

[cleanup_inventory.json](cleanup_inventory.json) inventories every file tracked at the starting cleanup commit, with SHA-256 and disposition. [source_imports.json](source_imports.json) records every companion-repository file and destination.

No pre-existing tracked files have been deleted or moved. Generated Python artifacts and the real data manifest remain pending specific removal/untracking approval. Pre-existing ALBEF notebook outputs and results remain preserved until their scientific relevance is reviewed. Five newly imported companion notebooks have generated outputs stripped. They are not automatically accepted as thesis evidence.

Root packages and historical script/config paths remain canonical during baseline validation. No model or trainer has been refactored. A missing import reference does not establish that a script is unused; launchers and notebooks must also be checked.
