import concepts

example_name = {'yelp_polarity': 'text'}
concepts_from_labels = { 'yelp_polarity': ["negative","positive"]}
class_num = {'yelp_polarity': 2}

# Config for Roberta-Base baseline
finetune_epoch = {'yelp_polarity': 2}
finetune_mlp_epoch = { 'yelp_polarity': 3}

# Config for CBM training
concept_set = {'yelp_polarity': concepts.yelpp}
#cbl_epochs = { 'yelp_polarity': 3}

cbl_epochs = {'yelp_polarity': 2}
