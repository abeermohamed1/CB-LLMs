import concepts

example_name = {'yelp_polarity': 'text', 'hotelreview': 'text'}
concepts_from_labels = { 'yelp_polarity': ["negative","positive"], 'hotelreview': ["negative","positive"]}
class_num = {'yelp_polarity': 2, 'hotelreview': 2}

# Config for Roberta-Base baseline
finetune_epoch = {'yelp_polarity': 2, 'hotelreview': 2}
finetune_mlp_epoch = { 'yelp_polarity': 3,  'hotelreview': 3}

# Config for CBM training
concept_set = {'yelp_polarity': concepts.yelpp, 'hotelreview': concepts.hotelreview}
#cbl_epochs = { 'yelp_polarity': 3, 'hotelreview': 3}

cbl_epochs = {'yelp_polarity': 2, 'hotelreview': 2}

