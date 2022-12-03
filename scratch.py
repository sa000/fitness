def get_columns_to_drop(excercise: str):
    '''
    Get the columns to drop from the dataframe. Aim to help model learn faster with only relevant features
    '''
    if excercise=='kb_around_world':
        not_revelant_body_parts = ['NOSE', 'EYE', 'EAR', 'KNEE', 'ANKLE']
    #for each not relevant body part, remove the LEFT, RIGHT and SCORE columns from the BODYPART columns
    columns_to_drop = []
    for body_part in not_revelant_body_parts:
        print(body_part)
        for column in [body_part+'_x', body_part+'_y', body_part+'_score']:
            columns_to_drop.append(column)
    return columns_to_drop

print(get_columns_to_drop('kb_around_world'))