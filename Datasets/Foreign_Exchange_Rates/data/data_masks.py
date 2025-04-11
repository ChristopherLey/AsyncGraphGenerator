from datetime import datetime


def convert_index_to_datetime(idx):
    return datetime(year=(2000 + int(idx // 12)), month=int(idx % 12), day=1)


def generate_masks():
    import random

    rnd_index = [x for x in range(1, 12 * 20, 2)]
    shuffled_list = rnd_index.copy()
    random.shuffle(shuffled_list)
    test_val = int(len(shuffled_list) * 0.4)
    train_list = shuffled_list[test_val:]
    rest_list = shuffled_list[:test_val]
    val_list = rest_list[: int(len(rest_list) * 0.5)]
    test_list = rest_list[int(len(rest_list) * 0.5) :]
    train_list = [convert_index_to_datetime(x) for x in train_list]
    train_list.sort()
    val_list = [convert_index_to_datetime(x) for x in val_list]
    val_list.sort()
    test_list = [convert_index_to_datetime(x) for x in test_list]
    test_list.sort()
    return train_list, val_list, test_list


training_masks = [
    datetime(2000, 3, 1, 0, 0),
    datetime(2000, 7, 1, 0, 0),
    datetime(2001, 1, 1, 0, 0),
    datetime(2001, 5, 1, 0, 0),
    datetime(2001, 7, 1, 0, 0),
    datetime(2001, 9, 1, 0, 0),
    datetime(2001, 11, 1, 0, 0),
    datetime(2002, 3, 1, 0, 0),
    datetime(2002, 7, 1, 0, 0),
    datetime(2002, 9, 1, 0, 0),
    datetime(2002, 11, 1, 0, 0),
    datetime(2003, 1, 1, 0, 0),
    datetime(2003, 7, 1, 0, 0),
    datetime(2003, 9, 1, 0, 0),
    datetime(2003, 11, 1, 0, 0),
    datetime(2004, 1, 1, 0, 0),
    datetime(2004, 9, 1, 0, 0),
    datetime(2004, 11, 1, 0, 0),
    datetime(2005, 1, 1, 0, 0),
    datetime(2005, 3, 1, 0, 0),
    datetime(2005, 9, 1, 0, 0),
    datetime(2006, 1, 1, 0, 0),
    datetime(2006, 9, 1, 0, 0),
    datetime(2007, 1, 1, 0, 0),
    datetime(2007, 5, 1, 0, 0),
    datetime(2007, 11, 1, 0, 0),
    datetime(2008, 5, 1, 0, 0),
    datetime(2008, 7, 1, 0, 0),
    datetime(2008, 9, 1, 0, 0),
    datetime(2009, 1, 1, 0, 0),
    datetime(2009, 5, 1, 0, 0),
    datetime(2009, 9, 1, 0, 0),
    datetime(2009, 11, 1, 0, 0),
    datetime(2010, 1, 1, 0, 0),
    datetime(2010, 9, 1, 0, 0),
    datetime(2010, 11, 1, 0, 0),
    datetime(2011, 1, 1, 0, 0),
    datetime(2011, 3, 1, 0, 0),
    datetime(2011, 7, 1, 0, 0),
    datetime(2011, 9, 1, 0, 0),
    datetime(2011, 11, 1, 0, 0),
    datetime(2012, 1, 1, 0, 0),
    datetime(2012, 3, 1, 0, 0),
    datetime(2012, 7, 1, 0, 0),
    datetime(2013, 5, 1, 0, 0),
    datetime(2013, 11, 1, 0, 0),
    datetime(2014, 1, 1, 0, 0),
    datetime(2014, 3, 1, 0, 0),
    datetime(2014, 7, 1, 0, 0),
    datetime(2014, 11, 1, 0, 0),
    datetime(2015, 1, 1, 0, 0),
    datetime(2015, 3, 1, 0, 0),
    datetime(2015, 5, 1, 0, 0),
    datetime(2015, 7, 1, 0, 0),
    datetime(2016, 3, 1, 0, 0),
    datetime(2016, 7, 1, 0, 0),
    datetime(2016, 11, 1, 0, 0),
    datetime(2017, 1, 1, 0, 0),
    datetime(2017, 3, 1, 0, 0),
    datetime(2017, 5, 1, 0, 0),
    datetime(2017, 7, 1, 0, 0),
    datetime(2017, 9, 1, 0, 0),
    datetime(2017, 11, 1, 0, 0),
    datetime(2018, 3, 1, 0, 0),
    datetime(2018, 7, 1, 0, 0),
    datetime(2018, 9, 1, 0, 0),
    datetime(2018, 11, 1, 0, 0),
    datetime(2019, 1, 1, 0, 0),
    datetime(2019, 3, 1, 0, 0),
    datetime(2019, 5, 1, 0, 0),
    datetime(2019, 9, 1, 0, 0),
    datetime(2019, 11, 1, 0, 0),
]

validation_masks = [
    datetime(2000, 11, 1, 0, 0),
    datetime(2001, 3, 1, 0, 0),
    datetime(2002, 1, 1, 0, 0),
    datetime(2003, 3, 1, 0, 0),
    datetime(2004, 7, 1, 0, 0),
    datetime(2005, 7, 1, 0, 0),
    datetime(2006, 5, 1, 0, 0),
    datetime(2007, 9, 1, 0, 0),
    datetime(2008, 3, 1, 0, 0),
    datetime(2008, 11, 1, 0, 0),
    datetime(2009, 3, 1, 0, 0),
    datetime(2010, 3, 1, 0, 0),
    datetime(2010, 5, 1, 0, 0),
    datetime(2010, 7, 1, 0, 0),
    datetime(2012, 5, 1, 0, 0),
    datetime(2012, 9, 1, 0, 0),
    datetime(2013, 1, 1, 0, 0),
    datetime(2013, 9, 1, 0, 0),
    datetime(2014, 9, 1, 0, 0),
    datetime(2015, 9, 1, 0, 0),
    datetime(2016, 1, 1, 0, 0),
    datetime(2016, 5, 1, 0, 0),
    datetime(2016, 9, 1, 0, 0),
    datetime(2018, 1, 1, 0, 0),
]

test_masks = [
    datetime(2000, 1, 1, 0, 0),
    datetime(2000, 5, 1, 0, 0),
    datetime(2000, 9, 1, 0, 0),
    datetime(2002, 5, 1, 0, 0),
    datetime(2003, 5, 1, 0, 0),
    datetime(2004, 3, 1, 0, 0),
    datetime(2004, 5, 1, 0, 0),
    datetime(2005, 5, 1, 0, 0),
    datetime(2005, 11, 1, 0, 0),
    datetime(2006, 3, 1, 0, 0),
    datetime(2006, 7, 1, 0, 0),
    datetime(2006, 11, 1, 0, 0),
    datetime(2007, 3, 1, 0, 0),
    datetime(2007, 7, 1, 0, 0),
    datetime(2008, 1, 1, 0, 0),
    datetime(2009, 7, 1, 0, 0),
    datetime(2011, 5, 1, 0, 0),
    datetime(2012, 11, 1, 0, 0),
    datetime(2013, 3, 1, 0, 0),
    datetime(2013, 7, 1, 0, 0),
    datetime(2014, 5, 1, 0, 0),
    datetime(2015, 11, 1, 0, 0),
    datetime(2018, 5, 1, 0, 0),
    datetime(2019, 7, 1, 0, 0),
]
