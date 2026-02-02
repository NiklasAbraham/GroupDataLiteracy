## Novelty analysis

For efficient computation of movie novelty scores, Faiss library was used.
Due to a mismatch in numpy versions between Faiss and the rest of the codebase, a separated conda environment is required to run the novelty analysis scripts. To install the required dependencies, run the following command:

```bash
conda env create -f src/aab_analysis/novelty/faiss_env.yml
```

Then activate the conda environment:

```bash
conda activate faiss
```