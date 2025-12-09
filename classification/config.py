import concepts

example_name = {'yelp_polarity': 'text', 'rakkaalhazimi/hotel-review': 'text'}
concepts_from_labels = { 'yelp_polarity': ["negative","positive"], 'rakkaalhazimi/hotel-review': ["negative","positive"]}
class_num = {'yelp_polarity': 2, 'rakkaalhazimi/hotel-review': 2}

# Config for Roberta-Base baseline
finetune_epoch = {'yelp_polarity': 2, 'rakkaalhazimi/hotel-review': 2}
finetune_mlp_epoch = { 'yelp_polarity': 3,  'rakkaalhazimi/hotel-review': 3}

# Config for CBM training
concept_set = {'yelp_polarity': concepts.yelpp, 'rakkaalhazimi/hotel-review': concepts.hotelreview}
#cbl_epochs = { 'yelp_polarity': 3, 'rakkaalhazimi/hotel-review': 3}

cbl_epochs = {'yelp_polarity': 2, 'rakkaalhazimi/hotel-review': 2}



