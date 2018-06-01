import requests

def githubimport(user, repo, module):
   d = {}
   url = 'https://raw.githubusercontent.com/{}/{}/master/{}.py'.format(user, repo, module)
   r = requests.get(url).text
   exec(r, d)
   return d

lstm = githubimport('Perry2018', 'LSTM', 'LSTMPredict')

a = lstm.LSTMPredict()
