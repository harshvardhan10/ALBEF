# Cleanup inventory

[cleanup_inventory.json](cleanup_inventory.json) inventories every file tracked at the starting cleanup commit, with SHA-256 and disposition. [source_imports.json](source_imports.json) records every companion-repository file and destination.

No pre-existing tracked files have been deleted or moved. The user approved stopping tracking the 70 generated Python artifacts and data/mimic_cxr.json on 2026-10-22. Their local files are preserved and ignored. The JSON inventory records their original tracked state/hashes, not their current tracking status. Pre-existing ALBEF notebook outputs and results remain preserved until their scientific relevance is reviewed. Five newly imported companion notebooks have generated outputs stripped. They are not automatically accepted as thesis evidence.

Root packages and historical script/config paths remain canonical during baseline validation. No model or trainer has been refactored. A missing import reference does not establish that a script is unused; launchers and notebooks must also be checked.
