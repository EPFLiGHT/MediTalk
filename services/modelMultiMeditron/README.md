# MultiMeditron Service

Service multimodal pour MediTalk utilisant le modèle MultiMeditron du LiGHT lab (EPFL).

## 🎯 Capacités

### Actuel (Meditron-7B)
- ✓ Questions médicales textuelles
- ✗ Pas de support d'images

### Nouveau (MultiMeditron)
- ✓ Questions médicales textuelles (compatibilité complète)
- ✓ **Support d'images médicales** (X-rays, CT scans, MRI, etc.)
- ✓ **Questions multimodales** (texte + image)
- ✓ Utilisation du token spécial `<|reserved_special_token_0|>`

## 📋 Architecture

```
Port 5006: Meditron-7B (existant, inchangé)
Port 5009: MultiMeditron (nouveau, multimodal)
```

## 🛠️ Implémentation

Le service suit **exactement** le pattern d'inférence décrit dans le [README de MultiMeditron](https://github.com/EPFLiGHT/MultiMeditron):

### Code Pattern (from MultiMeditron README)

```python
# Load model
model = MultiModalModelForCausalLM.from_pretrained("path/to/trained/model")

# Setup collator
collator = DataCollatorForMultimodal(
    tokenizer=tokenizer,
    tokenizer_type="llama",
    modality_processors=model.processors(),
    modality_loaders={"image": loader},
    attachment_token_idx=attachment_token_idx,
    add_generation_prompt=True
)

# Create sample
sample = {
    "conversations": [{"role": "user", "content": question}],
    "modalities": [{"type": "image", "value": "path/to/image"}]
}

# Generate
batch = collator([sample])
outputs = model.generate(batch=batch, temperature=0.1)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
```

### Notre Implémentation

Le fichier `app.py` implémente ce pattern dans la fonction `generate_multimeditron_response()`.

## 🚀 Configuration

### Variables d'Environnement

Ajoutez dans `.env`:

```bash
# Base LLM model (tokenizer source)
BASE_LLM=meta-llama/Meta-Llama-3.1-8B-Instruct

# Trained MultiMeditron model path
MULTIMEDITRON_MODEL=/path/to/trained/multimeditron/model
```

### Mode Fallback

Si `MULTIMEDITRON_MODEL` n'est pas configuré, le service fonctionne en **mode fallback** :
- ✓ Le service démarre
- ✓ Les endpoints répondent
- ⚠️ Réponses placeholder au lieu de vraies inférences
- → Permet de tester l'architecture sans modèle

## 📡 API Endpoints

### 1. `/health` - Health Check
```bash
curl http://localhost:5009/health
```

Réponse:
```json
{
  "status": "partial|ready|not_ready",
  "service": "MultiMeditron Medical AI",
  "model_loaded": true,
  "tokenizer_loaded": true,
  "note": "..."
}
```

### 2. `/ask` - Questions textuelles (compatible Meditron)
```bash
curl -X POST "http://localhost:5009/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is hypertension?",
    "max_length": 512,
    "temperature": 0.7,
    "generate_audio": true,
    "voice": "tara"
  }'
```

### 3. `/ask_multimodal` - Questions multimodales (NOUVEAU!)
```bash
curl -X POST "http://localhost:5009/ask_multimodal" \
  -F "question=What abnormalities do you see in this chest X-ray?" \
  -F "image=@/path/to/xray.jpg" \
  -F "max_length=512" \
  -F "temperature=0.7" \
  -F "generate_audio=true" \
  -F "voice=tara"
```

## Installation

### Automatique (recommandé)
```bash
cd /mloscratch/users/teissier/MediTalk
./setup-multimeditron.sh
```

### Manuel

1. **Créer l'environnement virtuel:**
```bash
cd services/modelMultiMeditron
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Installer MultiMeditron:**
```bash
# Option A: Cloner dans le service (recommandé)
git clone https://github.com/epflight/MultiMeditron.git multimeditron
pip install -e ./multimeditron

# Option B: Installer depuis GitHub
pip install git+https://github.com/epflight/MultiMeditron.git
```

3. **Configurer le modèle dans `.env`:**
```bash
MULTIMEDITRON_MODEL=/path/to/your/trained/model
```

## Structure du service

```
services/modelMultiMeditron/
├── app.py                    # Service FastAPI
├── requirements.txt          # Dépendances Python
├── README.md                 # Ce fichier
├── venv/                     # Virtual environment (gitignored)
└── multimeditron/           # Code MultiMeditron (gitignored)
    └── src/
        └── multimeditron/
            ├── model/
            ├── train/
            └── ...
```

## Démarrage

### Mode Test (MultiMeditron seul)
```bash
cd services/modelMultiMeditron
source venv/bin/activate
python app.py
```

Le service démarrera sur `http://localhost:5009`

### Mode Production (tous les services)
```bash
./start-with-multimeditron.sh
```

Démarre:
- Meditron-7B sur port 5006 (backup)
- MultiMeditron sur port 5009 (nouveau)
- Tous les autres services

## Configuration

### Variables d'environnement (`.env`)

```bash
# Requis
HUGGINGFACE_TOKEN=your_token_here

# MultiMeditron
MULTIMEDITRON_MODEL=/path/to/model  # Optionnel, fallback si vide

# Autres (inchangés)
MEDITRON_MODEL=epfl-llm/meditron-7b
ORPHEUS_MODEL=canopylabs/orpheus-3b-0.1-ft
WHISPER_MODEL=tiny
```

## Modes de fonctionnement

### 1. Mode Fallback (pas de modèle configuré)
- Service démarre sans erreur
- Répond avec des messages placeholder
- Utile pour tester l'infrastructure

### 2. Mode Partial (tokenizer chargé, modèle non chargé)
- Tokenizer MultiMeditron fonctionnel
- Génération pas encore implémentée
- Permet de tester l'API

### 3. Mode Full (tout chargé)
- Modèle MultiMeditron complet
- Génération multimodale fonctionnelle
- Mode production

## Développement

### Structure du code

```python
# Chargement du modèle
from multimeditron.model.model import MultiModalModelForCausalLM

model = MultiModalModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    use_safetensors=True
).to(device)

# Utilisation du token spécial
ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"
```

### Points d'extension

1. **`startup_event()`**: Charger le modèle
2. **`ask_question()`**: Génération text-only
3. **`ask_multimodal_question()`**: Génération multimodale
4. **`generate_fallback_response()`**: Réponses de fallback

## Tests

### 1. Test de santé
```bash
curl http://localhost:5009/health | jq
```

### 2. Test text-only
```bash
curl -X POST "http://localhost:5009/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is diabetes?", "generate_audio": false}' | jq
```

### 3. Test multimodal
```bash
# Télécharger une image de test
wget https://example.com/chest_xray.jpg -O test_xray.jpg

# Envoyer la requête
curl -X POST "http://localhost:5009/ask_multimodal" \
  -F "question=Describe this X-ray" \
  -F "image=@test_xray.jpg" \
  -F "generate_audio=false" | jq
```

## Troubleshooting

### Service ne démarre pas
```bash
# Vérifier les logs
tail -f ../../logs/modelMultiMeditron.log

# Vérifier l'environnement virtuel
source venv/bin/activate
python -c "import torch; print(torch.__version__)"
```

### Modèle ne charge pas
```bash
# Vérifier le chemin
echo $MULTIMEDITRON_MODEL
ls -la $MULTIMEDITRON_MODEL

# Vérifier la RAM/GPU
nvidia-smi  # Si GPU
free -h     # RAM
```

### Import errors
```bash
# Réinstaller MultiMeditron
source venv/bin/activate
pip uninstall multimeditron
pip install git+https://github.com/epflight/MultiMeditron.git
```

## Migration depuis Meditron-7B

Le service MultiMeditron est **100% compatible** avec l'API Meditron existante via l'endpoint `/ask`.

Pour migrer progressivement:

1. **Phase 1**: Garder les deux (5006 et 5009)
2. **Phase 2**: Router certaines requêtes vers MultiMeditron
3. **Phase 3**: Utiliser MultiMeditron par défaut, Meditron en backup
4. **Phase 4**: Remplacer complètement (optionnel)

## Rollback

Si problème avec MultiMeditron:

```bash
# Arrêter tous les services
./stop-local.sh

# Redémarrer sans MultiMeditron
./start-local.sh

# Meditron-7B continue sur port 5006 (aucun impact)
```

## Références

- [MultiMeditron Repository](https://github.com/epflight/MultiMeditron)
- [Guide d'intégration](../../MULTIMEDITRON-INTEGRATION.md)
- [Documentation MediTalk](../../README.md)
