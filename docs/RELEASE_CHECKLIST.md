# Anonymous GitHub Release Checklist

1. Run `bash scripts/validate_release.sh`.
2. Confirm `.env` is ignored and only `.env.example` is staged.
3. Confirm no model, dataset, checkpoint, profile, raw repeat, or log is staged.
4. Create an empty repository from an anonymous account whose profile, avatar,
   email, organizations, and external links do not identify any author. Do not
   initialize the remote with a README, license, or template files.
5. Copy only the public-code directory to a clean location outside the parent
   research repository. Initialize a new history and configure identity only
   for this repository:

   ```bash
   git init -b main
   git config --local user.name "Anonymous Authors"
   git config --local user.email "ANONYMOUS_ACCOUNT_NOREPLY_EMAIL"
   git add .
   git diff --cached --check
   git status --short
   ```

6. Inspect the staged content, filenames, and commit identity:

   ```bash
   git grep -n -E '/home/|10\.[0-9]+\.|@[A-Za-z0-9._-]+'
   git diff --cached --name-only
   git commit -m "Initial code release"
   git log -1 --format=fuller
   ```

7. Add the empty anonymous remote and push the new one-commit history:

   ```bash
   git remote add origin git@github.com:ANONYMOUS_HANDLE/ANONYMOUS_REPOSITORY.git
   git push -u origin main
   ```

8. Never push the parent research workspace or its Git history. Check the
   public repository in a logged-out browser session for account, commit, issue,
   release, and workflow metadata that could identify an author.
9. From a fresh clone, rerun the validator and reference-figure command.

`bash scripts/package_release.sh` creates a validated tarball and prints its
SHA-256 checksum for an additional clean-room inspection.
