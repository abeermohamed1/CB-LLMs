import concepts

example_name = {'yelp_polarity': 'text', 'hotel_review': 'text'}
concepts_from_labels = { 'yelp_polarity': ["negative","positive"], 'hotel_review': ["negative","positive"]}
class_num = {'yelp_polarity': 2, 'hotel_review': 2}

# Config for Roberta-Base baseline
finetune_epoch = {'yelp_polarity': 2, 'hotel_review': 2}
finetune_mlp_epoch = { 'yelp_polarity': 3,  'hotel_review': 3}

# Config for CBM training
concept_set = {'yelp_polarity': concepts.yelpp, 'hotel_review': concepts.hotelreview}
#cbl_epochs = { 'yelp_polarity': 3, 'hotel_review': 3}

cbl_epochs = {'yelp_polarity': 2, 'hotel_review': 2}


