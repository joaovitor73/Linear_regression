name = "Model"
config = "Config1"

# Usando o operador +
combined = name + config
print(combined)  # Saída: Model+Config1

# Usando f-strings (Python 3.6+)
combined = f"{name}+{config}"
print(combined)  # Saída: Model+Config1

# Usando format()
combined = "{}+{}".format(name, config)
print(combined)  # Saída: Model+Config1
