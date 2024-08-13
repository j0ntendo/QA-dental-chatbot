import orjson
# remove the id 1 because the crawlwer messed up

def reformat_json_file(file_path):
    
    with open(file_path, 'rb') as file:
        data = orjson.loads(file.read())
    
    
    if data and data[0].get("id") == 1:
        data = data[1:]

    
    for i, item in enumerate(data, start=1):
        item["id"] = i

    
    reformatted_json = orjson.dumps(data, option=orjson.OPT_INDENT_2)
    
    
    with open(file_path, 'wb') as file:
        file.write(reformatted_json)


file_path = 'forum_data.json'


reformat_json_file(file_path)
