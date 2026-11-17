# Save the package to GitHub

1. Create a **new empty repository** in your GitHub account, preferably private until all four authors approve release.
2. Download and extract `Niger_Basin_RSASE_GitHub_Repository.zip` locally.
3. From the extracted directory run:

   ```bash
   git init
   git add README.md requirements.txt run.py .gitignore docs scripts tests
   git status
   git commit -m "Add Niger Basin RSASE reproducibility code"
   git branch -M main
   git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPO.git
   git push -u origin main
   ```

   Replace `YOUR-ACCOUNT/YOUR-REPO` with the actual newly created repository path.

4. Do not include private Drive exports or original GeoTIFFs. The provided `.gitignore` excludes common data formats but you must still review `git status` before committing.
5. Run the preflight check in Colab after cloning, and link the actual repository in the manuscript data/code availability statement only after the repository exists.
6. Consider issuing a release tag and Zenodo DOI after author approval, if the data/software licensing permits.
