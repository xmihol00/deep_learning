import re

file_reg_names= [("dropout_05.txt", "\\\\batch\\\\normalization,\\\\dropout 0.5"), ("dropout_04.txt", "\\\\batch\\\\normalization,\\\\dropout 0.4"),
                 ("dropout_03.txt", "\\\\batch\\\\normalization,\\\\dropout 0.3"),
                 ("l1_001.txt", "\\\\L1 0.01"), ("l1_0001.txt", "\\\\L1 0.001"), ("l1_00001.txt", "\\\\L1 0.0001"),
                 ("l2_001.txt", "\\\\L2 0.01"), ("l2_0001.txt", "\\\\L2 0.001"), ("l2_00001.txt", "\\\\L2 0.0001"),
                 ("l1l2_001.txt", "\\\\L1 0.01\\\\L2 0.01"), ("l1l2_0001.txt", "\\\\L1 0.001\\\\L2 0.001"), 
                 ("l1l2_00001.txt", "\\\\L1 0.0001\\\\L2 0.0001"),]

model_names = ["FC -- SP\\\\16-256\\\\10",  
               "FC -- MP\\\\16-256\\\\128, 10",  
               "FC -- MP\\\\32-512\\\\256, 10",  
               "VGG -- 2 B\\\\32-64\\\\256, 128, 10",  
               "VGG -- 3 B\\\\16-64\\\\128, 10",  
               "VGG -- 3 B\\\\32-128\\\\256, 10"]

row_end = "\\\\\n"
row_end_line = "\\\\\n\\hline\n"
percent = "\\%"

print("\\begin{center}", "\\begin{tabular}{ |c|c|c|c|c|c|c| }", sep="\n")
print("\\hline")
print("& ", end="")
for j in range(len(model_names)):
    print(f"\\thead{{{model_names[j]}}}", f"{row_end if j + 1 == len(model_names) else '& '}", end="")
print("\\hline")

for i, (file_name, reg_name) in enumerate(file_reg_names):
    if i == 6:
        print("\\end{tabular}", "\\end{center}", sep="\n")
        print("\n")
        print("\\begin{center}", "\\begin{tabular}{ |c|c|c|c|c|c|c| }", sep="\n")
        print("\\hline")
        print("& ", end="")
        for j in range(len(model_names)):
            print(f"\\thead{{{model_names[j]}}}", f"{row_end if j + 1 == len(model_names) else '& '}", end="")
        print("\\hline")

    file = open(file_name, "r").readlines()

    epoch = ""
    models = []
    current_model = []
    for line in file:
        match = re.match(r"^Epoch (\d+)/.*", line)
        if match:
            epoch = match.group(1)
            if epoch == '1':
                if len(current_model):
                    models.append(current_model[-3])
                    current_model = []
        else:
            loss_match = re.match(r".*?loss: (\d\.\d+).*", line)
            accuracy_match = re.match(r".*?accuracy: (0\.\d+).*", line)
            val_loss_match = re.match(r".*?val_loss: (\d\.\d+).*", line)
            val_accuracy_match = re.match(r".*?val_accuracy: (0\.\d+).*", line)
            current_model.append((
                float(loss_match.group(1)),
                float(val_loss_match.group(1)),
                float(accuracy_match.group(1)) * 100,
                float(val_accuracy_match.group(1)) * 100,
                epoch
            ))
    models.append(current_model[-3])

    print("\\thead{", reg_name, "} & ", end="", sep="")
    for j, model_name in enumerate(model_names):
        print("\\makecell{")
        print(f"Epoch: {models[j][4]} {row_end}", end="")
        print(f"{models[j][0]:.2f} {row_end}", end="")
        print(f"{models[j][1]:.2f} {row_end}", end="")
        print(f"{models[j][2]:.2f} {percent} {row_end}", end="")
        print(f"{models[j][3]:2.2f} {percent} {row_end}", end="")
        print("}", f"{row_end_line if j + 1 == len(model_names) else '& '}", end="")

print("\\end{tabular}", "\\end{center}", sep="\n")
