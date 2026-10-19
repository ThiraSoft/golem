# FLUX 3 Image Klein dans golem et krea2-mobile

## Pourquoi

Ricardo veut faire tourner FLUX 3 dans golem et s'en servir depuis krea2-mobile.
FLUX 3 Video n'a pas de poids publics (accès anticipé par API seulement, FLUX 3
Dev annoncé sans date). Les seuls poids ouverts de la famille sont **FLUX 3
Image Klein** (5,6B, 15 octobre 2026, licence FLUX Kommunity, sur HF derrière
acceptation). Décision de Ricardo : Klein dans golem maintenant, la vidéo
attendra FLUX 3 Dev.

Klein sait partir d'un prompt seul (t2i) et d'une ou plusieurs images de
référence (i2i, mi2i). C'est exactement ce que les modes Compose et Edit de
krea2-mobile attendent et que golem ne savait pas faire : Ricardo a demandé
« les deux », texte et image.

## Buts

- Paquet `flux3` dans golem, du prompt (et d'images de référence) à l'image,
  sur la carte (RX 9070 XT, 16 Go, Vulkan, coopmat), à la manière de `krea2` :
  transcrit de ComfyUI (commit `20b46065`, « Support Flux3-image ») et vérifié
  point par point contre lui.
- `cmd/flux3` et `golem-server -flux3 DIR` sur `/v1/images/generations`,
  avec références.
- krea2-mobile : choix du modèle (Krea 2 ou FLUX 3 Klein), modes Compose et
  Edit actifs avec FLUX 3.
- Perf mesurée (s/image à 1024², s/étape, temps encodeur/VAE), comparée à
  ComfyUI sur la même carte.

## Non-buts

- La vidéo (pas de poids). FLUX 3 Action (robotique).
- Le prompt upsampler Qwen3.5-9B livré avec Klein.
- Les LoRA FLUX 3 (aucune n'existe encore ; le mécanisme bypass de krea2 se
  rebranchera plus tard).
- Le Layout Prompt (boîtes englobantes).

## Anti-buts

- Une image différente de ComfyUI à l'œil pour le même prompt et la même
  graine. La référence fait foi, comme pour krea2.
- Casser krea2 ou le mobile existant (Krea 2 reste le défaut).
- Toucher au checkout ComfyUI de Ricardo (`/mnt/data/dev/ComfyUI`, branche
  `wip` avec le front mobile) : la référence tourne dans un worktree séparé.

## Le modèle (d'après ComfyUI)

- **Encodeur texte** : Qwen3-VL-4B, le même que Krea 2
  (`qwen3vl_4b_fp8_scaled.safetensors`, déjà sur la machine). Template
  instruct sans bloc think, padding à droite au multiple de 80 tokens avec
  151643 ; les positions rope des pads valent 1 ; masque d'attention sur les
  pads. Sorties lues à 8 couches `[4, 8, …, 32]`, mises côte à côte par
  token : 8 × 2560 = 20480. Le DiT lit aussi les pads.
- **DiT** : hidden 3072, 24 têtes de 128, `depth` blocs par mode (image,
  image_cond, txt) puis `depth_single_blocks` blocs joints ; chaque bloc est
  un bloc parallèle (LayerNorm sans affine modulé, q/k/v ou qkv fusionné,
  QK-RMSNorm + rope, attention, SwiGLU ratio 3, une seule porte). Rope 4 axes
  de 32 (t, ligne × 0,5, colonne × 0,5, l ; références à l = −1−i, texte sur
  l = index). Modulations distinctes par mode, texte et références à t = 0.
  Variante `flow_map` (Klein distillé) : `ref_time_in` ajoute l'embedding du
  sigma suivant à toutes les modulations ; alors les blocs early du texte et
  des références se recalculent à chaque étape. Sans flow_map, une fois par
  image.
- **VAE** : `flux2-vae.safetensors`, AutoencoderKL 32 canaux, ×8, batch norm
  latente, patch 2×2 → 128 canaux à ×16. Décodeur pour sortir l'image,
  encodeur pour les références.
- **Échantillonnage** : flow, shift 1,93586. Sampler, étapes et CFG par
  défaut : ceux des workflows BFL `flux3_klein_*_t2i.json` / `_i2i.json`.

## Fichiers

Depuis `black-forest-labs/flux-3-image-klein-comfy` vers
`/mnt/data/dev/ComfyUI/models/` : `diffusion_models/flux3-image-klein-bf16`,
`vae/flux2-vae`, et les workflows. Choix golem (mon appel) : lire le bf16 et le
passer en fp8 e4m3 avec une échelle par tenseur au chargement, pour réutiliser
le produit fp8 de krea2 (5,6 Go sur la carte) ; garder le chemin fp16 si la
mesure montre un écart visible avec ComfyUI.

## Laissé au constructeur

Découpe des shaders, fusion des noyaux, taille d'arène, ordre des
optimisations, tant que chaque étape est mesurée.
