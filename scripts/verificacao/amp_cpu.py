"""Roda src/train.py com AMP float16 FORÇADO em CPU (só para teste).
Troca, na memória, duas coisas do train.py:
  - usar_amp passa a valer em CPU;
  - o autocast usa float16 (o padrão de CPU seria bfloat16; na GPU é float16).
Todo o resto é o código real."""
import sys, os, re, runpy
repo = sys.argv[1]; sys.argv = ["src.train"] + sys.argv[2:]
sys.path.insert(0, repo); os.chdir(repo)
fonte = open("src/train.py").read()
n1 = fonte.count('usar_amp = args.amp and device.type == "cuda"')
fonte = fonte.replace('usar_amp = args.amp and device.type == "cuda"', 'usar_amp = args.amp')
n2 = fonte.count("torch.amp.autocast(device.type, enabled=usar_amp)")
fonte = fonte.replace("torch.amp.autocast(device.type, enabled=usar_amp)",
                      "torch.amp.autocast(device.type, dtype=torch.float16, enabled=usar_amp)")
assert n1 == 1 and n2 >= 3, (n1, n2)
print(f"[amp_cpu] trocas: usar_amp={n1}, autocast={n2}", flush=True)
codigo = compile(fonte, "src/train.py", "exec")
ns = {"__name__": "__main__", "__file__": "src/train.py"}
exec(codigo, ns)
