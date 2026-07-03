# -AVA-4-Sistemas-evolutivos-e-aplicados-a-rob-tica-H10322-

## Código Corrigido - Geração de Imagem com Stable Diffusion

```python
from diffusers import StableDiffusionPipeline
import torch
import time

# Carrega o modelo pré-treinado
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
torch_dtype=torch.float16)

# Move para GPU
pipe = pipe.to("cuda")

# Inicia cronometragem
ti = time.time()

# Gera a imagem
prompt = "gato na lua"
image = pipe(prompt).images[0]

# Finaliza cronometragem
tf = time.time()
torch_type = tf - ti

# Salva e exibe
image.save("output.png")
image.show()

print(f"Tempo de execução: {torch_type:.2f} segundos")
```

### Principais Correções:
1. ✅ Moveu `ti = time.time()` para **antes** da geração
2. ✅ Moveu `tf = time.time()` para **depois** da geração
3. ✅ Adicionado `import time` 
4. ✅ Adicionado print para exibir o tempo de execução
