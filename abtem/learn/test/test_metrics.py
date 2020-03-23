from abtem.points import Points

cell = [1,1]
label_positions = [[-.5, .5], [.5,.5], [.5,.9], [.5, -.1], [.9,.9]]
detected_positions = [[1.5, 1.5], [.5,.5], [.5, 1.1], [.5,.1], [.1,.1]]

label = Points(label_positions, cell)
detected = Points(detected_positions, cell)

evaluator = Evaluator(.1)
evaluator.find_pairs(label, detected, inside_cell(label))

print(evaluator.get_true_positives().positions)
print(evaluator.get_false_positives().positions)
print(evaluator.get_false_negatives().positions)
