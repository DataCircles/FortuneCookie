import gpt_2_simple as gpt2
import tensorflow as tf
import pandas as pd

from library.common import _random_prefix

gpt2.download_gpt2(model_name="124M")
file_name = "training_data/data.txt"

def gpt2_finetune(data, steps, run_name):

    tf.reset_default_graph()
    sess = gpt2.start_tf_sess()

    return gpt2.finetune(sess,
                         dataset=data,
                         steps=steps,
                         run_name=run_name,
                         model_name='124M',
                         restore_from='fresh',
                         print_every=10,
                         sample_every=200,
                         save_every=500
                         )


def text_generator(model_name, prefix, temperature, **kwargs):

    tf.reset_default_graph()
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name=model_name)

    return gpt2.generate(sess,
                         length=20,
                         temperature=temperature,
                         prefix=prefix,
                         nsamples=5,
                         batch_size=5
                         )

data = pd.read_csv(file_name)
corpus = data['Fortune Cookie Quotes']
start_word = _random_prefix(corpus)

gpt2_finetune(data=file_name, steps=100, run_name='run1')
text_generator(model_name='run1', prefix=start_word, temperature=1)
