# Push Code to GitHub

## Quick Steps

### 1. Initialize Git (if not already)

```bash
cd /Users/kalyaanrao/finalVersion
git init
```

### 2. Add All Files

```bash
git add .
```

### 3. Commit

```bash
git commit -m "Initial commit: SmolLM training pipeline with Muon+AdamW, WSD scheduler, and W&B integration"
```

### 4. Add Remote

```bash
git remote add origin https://github.com/kalyaannnn/smolLM.git
```

### 5. Push

```bash
git branch -M main
git push -u origin main
```

---

## If You Need to Authenticate

```bash
# Use GitHub CLI
gh auth login

# Or use personal access token
git remote set-url origin https://YOUR_TOKEN@github.com/kalyaannnn/smolLM.git
```

---

## Verify Push

Visit: https://github.com/kalyaannnn/smolLM

You should see all your files!

