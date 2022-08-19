import yaml

with open("examples.yml", 'r') as stream:
    data = yaml.safe_load(stream)

print(data)
