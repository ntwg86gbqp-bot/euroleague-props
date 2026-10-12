# Euroleague Props

Ειδικά στοιχήματα παικτών της Euroleague: ιστορικό, μοντέλο πιθανοτήτων, αξία.

Τρέχει στο Streamlit Community Cloud. Τα δεδομένα αποθηκεύονται αυτόματα στον κλάδο `data`
αυτού του αποθετηρίου (μην τον σβήσεις). Χρειάζονται τα secrets:

```toml
GITHUB_TOKEN = "github_pat_…"        # fine-grained, μόνο αυτό το αποθετήριο, Contents: Read and write
DATA_REPO = "χρήστης/euroleague-props"
```
