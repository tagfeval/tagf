import json

# create a dictionary to store your twitter credentials

twitter_cred = dict()

# Enter your own consumer_key, consumer_secret, access_key and access_secret
# Replacing the stars ("********")

twitter_cred['CONSUMER_KEY'] = 'vIcGwT81mg2wKsgxNZ2O9LoU7'
twitter_cred['CONSUMER_SECRET'] = 'CULG1qLjCK6MlisDDgdRCyzsfDHPm5YyXzm1h7ac4Vs3Q9FkiO'
twitter_cred['ACCESS_KEY'] = '1112716803938140165-h15D5yK93YnBVVnYh5MEoucbVSKKU1'
twitter_cred['ACCESS_SECRET'] = '3vMwXba2RZFmkbfMbED2HsUpekDeJTULM3tt9bGrgl7sm'

# Save the information to a json so that it can be reused in code without exposing
# the secret info to public

with open('twitter_credentials.json', 'w') as secret_info:
    json.dump(twitter_cred, secret_info, indent=4, sort_keys=True)
