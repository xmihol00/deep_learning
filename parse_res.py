import re

file = open("basic.txt", "r").readlines()

models = []
current_model = []
for line in file:
    match = re.match(r"^Epoch (\d+)/.*", line)
    if match:
        if match.group(1) == '1':
            if len(current_model):
                models.append(current_model)
                current_model = []
    else:
        match = re.match(r".*?val_accuracy: (0\.\d+).*", line)
        current_model.append(float(match.group(1)) * 100)
models.append(current_model)

max_len = 0
for model_res in models:
    res_len = len(model_res)
    if res_len > max_len:
        max_len = res_len

for model_res in models:
    while len(model_res) < max_len:
        model_res.append("-----")

model_names = ["FC -- SP\\\\16-256", "FC -- MP\\\\16-256", "FC -- MP\\\\32-512", "VGG -- 2 B\\\\32-64", "VGG -- 3 B\\\\16-64", "VGG -- 3 B\\\\32-128"]

print("\\begin{center}", "\\begin{tabular}{ |c|c|c|c|c|c| }", sep="\n")

row_end = "\\\\\n"
percent = "\\%"

print("\\hline")
print("& ", end="")
for i in range(len(models)):
    print(f"\\thead{{{model_names[i]}}}", f"{row_end if i + 1 == len(models) else '& '}", end="")
print("\\hline")


for i in range(max_len):
    print(f"   \\textbf{{Epoch {i + 1}}} & ", end="")
    for j, model_res in enumerate(models):
        print(f"{'{:2.2f} {}'.format(model_res[i], percent) if isinstance(model_res[i], float) else model_res[i]}", 
              f"{row_end if j + 1 == len(models) else '& '}", end="")

print("\\hline")
print("\\end{tabular}", "\\end{center}", sep="\n")
