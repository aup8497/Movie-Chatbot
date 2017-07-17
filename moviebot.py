# --------------------------------------------------------------------------------------------
# import statements
# --------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import re
import nltk
import sys
import time
import math
import random
import os
import pickle

# --------------------------------------------------------------------------------------------
# initializing constants and flags
# --------------------------------------------------------------------------------------------
TEST_MODE = False
TRAIN_MODE = False
if(len(sys.argv)<2):
	print("enter whether to test or train")
	sys.exit(0)
else:
	if sys.argv[1] == '--train':
	  TRAIN_MODE = True
	elif sys.argv[1] == '--test-interactive':
	  TEST_MODE = True

MAX_INPUT_LENGTH = 10
BATCH_SIZE = 10
LEARNING_RATE = 0.001
SAVE_EVERY_N_STEP = 200
STEPS_PER_CKPT = 10
HIDDEN_SIZE = 256
NUM_LAYERS = 2
EMBEDDING_SIZE = 32
SAVE_FILE = 'moviebot.save'
NUM_STEPS_FILE = '.numsteps'


# --------------------------------------------------------------------------------------------
# creating the training samples
# --------------------------------------------------------------------------------------------
with open("vocab.pkl", "rb") as fp:   # Unpickling
  vocab = pickle.load(fp)

with open('training_samples.pkl','rb') as fp:
  training_samples = pickle.load(fp)

def getID(word):
	wid = vocab["word2id"].get(word, 3)  # returning 3 else, because 3 stands for unknown
	return wid

def sen2enco(sentence):
  return [getID(word) for word in nltk.word_tokenize(sentence)[:MAX_INPUT_LENGTH]]  


def createBatch(samples):
  batch_size = len(samples)
  enco_seqs = []
  deco_seqs = []
  target_seqs = []
  init_weights = []
  for i in range(batch_size):
    sample = samples[i]
    enco_seqs.append(list(reversed(sample[0])))  # Reverse inputs (and not outputs),as defined on the original seq2seq paper
    deco_seqs.append([getID('<go>')] + sample[1] + [getID('<eos>')])  # Add the <go> and <eos> tokens
    target_seqs.append(deco_seqs[-1][1:])  # Same as decoder, but shifted to the left (ignore the <go>)
    enco_seqs[i] = [getID('<pad>')] * (MAX_INPUT_LENGTH  - len(enco_seqs[i])) + enco_seqs[i]
    init_weights.append([1.0] * len(target_seqs[i]) + [0.0] * (MAX_INPUT_LENGTH + 2 - len(target_seqs[i])))
    deco_seqs[i] = deco_seqs[i] + [getID('<pad>')] * (MAX_INPUT_LENGTH + 2 - len(deco_seqs[i]))
    target_seqs[i]  = target_seqs[i]  + [getID('<pad>')] * (MAX_INPUT_LENGTH + 2 - len(target_seqs[i]))
  enco_seqs = np.asarray(enco_seqs).T.tolist()
  # enco_seqs is encoded sentence i.e [34,212,673,4,135]
  deco_seqs = np.asarray(deco_seqs).T.tolist()
  target_seqs = np.asarray(target_seqs).T.tolist()
  init_weights = np.asarray(init_weights).T.tolist()

  return (enco_seqs, deco_seqs, target_seqs, init_weights)


print('Creating training batches..')
training_batches = [createBatch(training_samples[k:k+BATCH_SIZE]) for k in range(0, len(training_samples), BATCH_SIZE)]
# here in the above statement the training samples are different each time 
print('Created ' + str(len(training_batches)) + ' batches. (total steps for one epoch)')
print('Training samples created.')


# --------------------------------------------------------------------------------------------
# creating the model
# --------------------------------------------------------------------------------------------
print('Creating model...')
network_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
network_cell = tf.contrib.rnn.MultiRNNCell([network_cell] * NUM_LAYERS, state_is_tuple=True)

placeh_encoder_inputs  = [tf.placeholder(tf.int32,   [None, ]) for _ in range(MAX_INPUT_LENGTH)]  # Batch size * sequence length * input dim
placeh_decoder_inputs  = [tf.placeholder(tf.int32,   [None, ], name='inputs') for _ in range(MAX_INPUT_LENGTH + 2)]  # Same sentence length for input and output (Right ?)
placeh_decoder_targets = [tf.placeholder(tf.int32,   [None, ], name='targets') for _ in range(MAX_INPUT_LENGTH + 2)]
placeh_decoder_weights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(MAX_INPUT_LENGTH + 2)]

decoder_outputs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
  placeh_encoder_inputs,
  placeh_decoder_inputs,
  network_cell,
  len(vocab["word2id"]),
  len(vocab["word2id"]),
  embedding_size=EMBEDDING_SIZE,
  output_projection=None,
  feed_previous=bool(TEST_MODE)
)

# --------------------------------------------------------------------------------------------
# Test mode
# --------------------------------------------------------------------------------------------
if TEST_MODE:
  outputs_prob = decoder_outputs
else:
  loss_function = tf.contrib.legacy_seq2seq.sequence_loss(
    decoder_outputs,
    placeh_decoder_targets,
    placeh_decoder_weights,
    len(vocab["word2id"]),
    softmax_loss_function = None
  )
  opt = tf.train.AdamOptimizer(
    learning_rate=LEARNING_RATE,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08
  )
  min_loss_opt = opt.minimize(loss_function)

print('Model created.')
## 3. TRAINING MODEL

print('Starting to train model...')
sess = tf.Session()
sess.run(tf.initialize_all_variables())
saver_obj = tf.train.Saver()
rest_epochs = 0

#Does previous model exist?
if os.path.exists(SAVE_FILE + '.index'):
  print('Previous model exists. Restoring model...')
  saver_obj.restore(sess, './' + SAVE_FILE)
  file_epochs = ""
  with open(NUM_STEPS_FILE, 'r') as f:
    file_epochs = f.read()
  rest_epochs = int(file_epochs)
  print('Model restored.')
else:
  print('No previously existing model found. Using new...')

step_time, glob_loss, loss = 0.0, 0.0, 0.0

current_step = 0
if os.path.exists(SAVE_FILE + '.index'):
  current_step = rest_epochs



# --------------------------------------------------------------------------------------------
# testing the input from the user
# --------------------------------------------------------------------------------------------
if TEST_MODE:
  feed_dict = {}

  while True:
    quest = input('YOU(enter "exit" to exit): ')
    if quest == 'exit':
      break
    enco = sen2enco(quest)
    enco_seq2, _, _, _ = createBatch([[enco, []]])
    for i in range(MAX_INPUT_LENGTH):
      feed_dict[placeh_encoder_inputs[i]] = enco_seq2[i]
    feed_dict[placeh_decoder_inputs[0]] = [getID('<go>')]

    ops = (outputs_prob,)
    output_itrv = sess.run(ops[0], feed_dict)
    rep_seq = [np.argmax(x) for x in output_itrv]
    sentence = []
    for wordId in rep_seq:
      if wordId == getID('<eos>'):  # End of generated sentence
        break
      elif wordId != getID('<pad>') and wordId != getID('<go>'):
        sentence.append(vocab["id2word"][wordId])
    respon = ' '.join(sentence)
    print("BOT: " + respon)
  print("Bye.")

## Training the model
if TRAIN_MODE:
  try:
    num_epochs = 0
    while True:
      random.shuffle(training_batches)
      for batch in training_batches:
        start_time = time.time()
        feed_dict = {}
        enco_seqs, deco_seqs, target_seqs, init_weights = batch
        for i in range(MAX_INPUT_LENGTH):
          feed_dict[placeh_encoder_inputs[i]] = enco_seqs[i]
        for i in range(MAX_INPUT_LENGTH + 2):
          feed_dict[placeh_decoder_inputs[i]] = deco_seqs[i]
          feed_dict[placeh_decoder_targets[i]] = target_seqs[i]
          feed_dict[placeh_decoder_weights[i]] = init_weights[i]

        ops = (min_loss_opt, loss_function)
        _, loss = sess.run(ops, feed_dict)

        glob_loss += loss / STEPS_PER_CKPT
        step_time += (time.time() - start_time) / STEPS_PER_CKPT
        current_step += 1

        if current_step % SAVE_EVERY_N_STEP:
          saver_obj.save(sess, SAVE_FILE)

        if current_step % STEPS_PER_CKPT == 0:
          perplexity = math.exp(float(glob_loss))
          print ("global step %d, step-time %.2f, perplexity "
                   "%.2f" % (current_step, step_time, perplexity))
          step_time, glob_loss = 0.0, 0.0
      
      num_epochs += 1
      print("Finished epoch " + str(num_epochs) + " of training.")
  except (KeyboardInterrupt, SystemExit):
    print('Exiting training. Saving model...')
    saver_obj.save(sess, SAVE_FILE)
    with open(NUM_STEPS_FILE, 'w') as f:
      f.write(str(current_step))
    sess.close()
    print('Training Done.')
