import os
import sys
import glob


starter3_exists = os.path.exists('starter3.py')
starter_exists = os.path.exists('starter.py')

if starter3_exists and starter_exists:
  sys.stderr.write('ERROR: You have both starter.py and starter3.py. You must delete one\n')
  exit(1)

try:
  import urllib.request as urllib2
except:
  import urllib2

def execute(cmd, envdict=None):
  if envdict is None: envdict = {}
  # Set environment variables.
  for k, v in envdict.items():
    os.environ[k] = str(v)
  # Execute.
  os.system(cmd)
  # Unset environment variables.
  for k in envdict.keys():
    os.environ.pop(k)

def fetch(file_path):
  url = 'http://sami.haija.org/cs544/a5/' + file_path.split('/')[-1]
  ff = urllib2.urlopen(url)
  with open(file_path, 'wb') as output:
    output.write(ff.read())

def fetch_and_compile(py, file_path):
  fetch(file_path)
  # Compile
  os.system('%s -m py_compile %s' % (py, file_path))
  os.remove(file_path)
  parts = file_path.split('/')
  filename = parts[-1].split('.py')[0]  # without .py
  search_pattern = parts[:-1] + ['__pycache__', filename + '.cpython*']
  search_pattern = os.path.join(*search_pattern)
  matching_files = glob.glob(search_pattern)
  if not matching_files:
    sys.stderr.write('Error: cannot find file matching ' + search_pattern + '\n')
    exit(0)
  elif len(matching_files) > 1:
    sys.stderr.write('Error: found multiple files matching ' + search_pattern + '. Delete them and retry\n')
    exit(0)
  return matching_files[0]

if len(sys.argv) > 1:
  # Fetch and run
  py = sys.argv[1]
  if starter3_exists:
    st = 'starter3'
  else:
    st = 'starter'
  # Fetch and compile.
  train_test = fetch_and_compile(py, 'grader/run_model.py')
  autograder = fetch_and_compile(py, 'grader/test_cases.py')
elif starter3_exists:
  py = 'python3'
  st = 'starter3'
  train_test = 'grader/run_model.cpython-36.pyc'
  autograder = 'grader/test_cases.cpython-36.pyc'
elif starter_exists:
  py = 'python'
  st = 'starter'
  train_test = 'grader/run_model.pyc'
  autograder = 'grader/test_cases.pyc'
else:
  sys.stderr.write('ERROR: Did not find neither starter.py nor starter3.py. You must have one\n')
  exit(1)



if not os.path.exists('grader'):
  os.makedirs('grader')

if not os.path.exists(train_test):
  fetch(train_test)

if not os.path.exists(autograder):
  fetch(autograder)

execute('echo Running test cases...')

execute('%s %s' % (py, autograder), {'STARTER': st})
txt = open('results.txt').read()
password = open('grader/password.txt').read()
email = open('grader/email.txt').read()
os.remove('grader/email.txt')
os.remove('grader/password.txt')
os.remove('results.txt')
os.remove('tagged_file1.txt')
os.remove('tagged_file2.txt')

print(txt)

lines = txt.split('\n')
if not lines[-1]:
  lines = lines[:-1]
last_line = lines[-1]
test_cases_score = float(last_line.split(',')[1])
# print('test_cases_score {}'.format(test_cases_score))

# Japenese
execute(
    '%s %s 1' % (py, train_test),
    {
      'LANGUAGE': 'japanese',
      'OUTPUT_ACCURACY_FILE': 'grader/japanese.txt',
      'STARTER': st,
      'MAX_TRAIN_TIME': 40000})
japanese_accuracy = float(open('grader/japanese.txt').read())
# japanese_accuracy = 0.947785
# print('Jap {}'.format(japanese_accuracy))
os.remove('grader/japanese.txt')

# Italian
execute(
    '%s %s 1' % (py, train_test),
    {
      'LANGUAGE': 'italian',
      'OUTPUT_ACCURACY_FILE': 'grader/italian.txt',
      'STARTER': st,
      'MAX_TRAIN_TIME': 120000})
italian_accuracy = float(open('grader/italian.txt').read())
# italian_accuracy = 0.954614
# print('Ital {}'.format(italian_accuracy))
os.remove('grader/italian.txt')

# Secret
execute('echo Be patient... Training secret language and sending predictions to server ...')
execute(
    '%s %s 0' % (py, train_test),
    {
      'LANGUAGE': 'surprise',
      'OUTPUT_PREDICTIONS_FILE': 'grader/preds.json',
      'STARTER': st,
      'MAX_TRAIN_TIME': 1200000})
surprise_preds = open('grader/preds.json').read()
# fja = open('test_j.txt', 'w')
# for i in surprise_preds:
#     fja.write(str(i)+'\n')
os.remove('grader/preds.json')


def accuracy_to_score(accuracy):
    if accuracy < 0.5:
      return 0
    elif accuracy > 0.85:
      return 30 + 30.0 * (accuracy - 0.85) / 0.075
    else:
      return 30.0 * (accuracy - 0.5) / 0.35

final_score = max(test_cases_score, 0) + (accuracy_to_score(japanese_accuracy) + accuracy_to_score(italian_accuracy))/2.0

if final_score < 20:
  final_score += 10

# Post the grades
import requests
import json

payload = json.dumps({
    'student': os.environ.get('VOC_EMAIL_ID', email),
    'surprise_preds': surprise_preds,
    'japanese': japanese_accuracy,
    'italian': italian_accuracy,
    'test_cases': test_cases_score,
    'score': final_score,
    'starter_name': st,
    #'starter_code': open(st + '.py').read(),
})
import hashlib
token = hashlib.md5(str(password + payload + 'emailer').encode("utf-8")).hexdigest()[:16]

HOST = 'http://haijaorg.appspot.com'

URI = HOST + '/cs544seq2seq19sp'

response = requests.post(URI, data='%s,%s' % (token, payload))
print('All done')
