# codeing: utf-8

import argparse
import load_data
import copy
import sklearn.metrics
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from mxnet import autograd
from model import CNNTextClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, help='File containing file representing the input TRAINING data')
parser.add_argument('--val_file', type=str, help='File containing file representing the input VALIDATION data', default=None)
parser.add_argument('--test_file', type=str, help='File containing file representing the input TEST data', default=None)
parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
parser.add_argument('--optimizer', type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
parser.add_argument('--batch_size', type=int, help='Training batch size', default=16)
parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0.5)
parser.add_argument('--embedding_source', type=str, default='glove.twitter.27B.100d', help='Pre-trained embedding source name')
parser.add_argument('--fix_embedding', action='store_true', help='Fix embedding vectors instead of fine-tuning them', default=False)
args = parser.parse_args()

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)


def train_classifier(vocabulary, data_train, data_val, data_test, ctx=mx.cpu()):

    # set up the data loaders for each data source
    print('...preparing and loading data...\n')
    train_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    val_dataloader   = mx.gluon.data.DataLoader(data_val, batch_size=args.batch_size, shuffle=True)
    test_dataloader = mx.gluon.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=True)
    
    emb_input_dim, emb_output_dim = vocabulary.embedding.idx_to_vec.shape
    model = CNNTextClassifier(emb_input_dim, emb_output_dim)

    model.initialize(ctx=ctx)  # initialize model parameters on the context ctx
    model.embedding.weight.set_data(vocabulary.embedding.idx_to_vec) # set the embedding layer parameters to the pre-trained embedding in the vocabulary

    if args.fix_embedding:
        print('FIX EMBEDDINGS')
        model.embedding.collect_params().setattr('grad_req', 'null')

    # model.hybridize() ## OPTIONAL for efficiency - perhaps easier to comment this out during debugging

    print('words similar to "earthquake" (prior to training):')
    print(get_knn(vocab, 10, 'earthquake'))
    print()
    print('words similar to "emergency":')
    print(get_knn(vocab, 10, 'emergency'))
    print()
    print('words similar to "flood":')
    print(get_knn(vocab, 10, 'flood'))
    print()
    print('words similar to "disaster":')
    print(get_knn(vocab, 10, 'disaster'))
    print()

    print('...training model...\n')
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})
    # for epoch in range(args.epochs):
    for epoch in range(args.epochs):
        epoch_cum_loss = 0
        for i, (label, data) in enumerate(train_dataloader):    # loop through batches
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = model(data)    # calls hybrid_forward
                loss = loss_fn(output, label).mean()    # get the average loss over the batch
            loss.backward()
            trainer.step(label.shape[0])  # update weights
            epoch_cum_loss += loss.asscalar()  # convert mx.nd.array value back to Python float
        val_acc, _, _, _, _ = evaluate(model, val_dataloader, False)
        # # display and/or collect validation accuracies after each training epoch
        # print('epoch {} val accuracy: '.format(epoch), val_acc)
        # print('epoch {} cum_loss: '.format(epoch), epoch_cum_loss)
        # print('~~~~~~~~~~~~~~~~~~~~~~')

    print('...classifying test data...\n')
    test_acc, test_prec, test_rec, test_f1, test_ap = evaluate(model, test_dataloader, True)
    print('test accuracy: ', test_acc)
    print('test precision: ', test_prec)
    print('test recall: ', test_rec)
    print('test F1: ', test_f1)
    print('test average precision score: ', test_ap)
    print()

    if not args.fix_embedding:
        # get new copy of Vocab with updated embedding weights attached
        updated_vocab = copy.deepcopy(vocab)
        token_emb = nlp.embedding.TokenEmbedding(allow_extend=True)
        token_emb.__setitem__(vocab.idx_to_token, model.embedding.weight.data())
        updated_vocab.set_embedding(token_emb)

        print('words similar to "earthquake" (after fine-tuning embeddings):')
        print(get_knn(updated_vocab, 10, 'earthquake'))
        print()
        print('words similar to "emergency":')
        print(get_knn(updated_vocab, 10, 'emergency'))
        print()
        print('words similar to "flood":')
        print(get_knn(updated_vocab, 10, 'flood'))
        print()
        print('words similar to "disaster":')
        print(get_knn(updated_vocab, 10, 'disaster'))
        print()


def evaluate(model, dataloader, is_test, ctx=mx.cpu()):
    """
    Get predictions on the dataloader items from model.
    correct prediction for 'Relevant' will approximate [1, 0];
    correct prediction for 'Not Relevant' will approximate [0, 1]
    Return metrics (accuracy, etc.)
    """

    labels = []  # store the ground truth labels
    predicted = []
    scores = []  # store the predictions/scores from the model
    for i, (label, data) in enumerate(dataloader):  # loop through batches
        # model inference
        out = mx.nd.softmax(model(data))

        # You'll then need to go over each item in the batch (or use array ops) as:
        for j in range(out.shape[0]):   # out.shape[0] refers to the batch size
            gold_index = int(mx.ndarray.argmax(label[j], axis=0).asscalar())
            labels.append(gold_index)
            predicted_index = int(mx.ndarray.argmax(out[j], axis=0).asscalar())
            predicted.append(predicted_index)
            scores.append(out[j][1].asscalar())

    acc = sklearn.metrics.accuracy_score(labels, predicted)
    precision_score = sklearn.metrics.precision_score(labels, predicted)
    recall_score = sklearn.metrics.recall_score(labels, predicted)
    f1 = sklearn.metrics.f1_score(labels, predicted)
    average_precision = sklearn.metrics.average_precision_score(labels, scores)
    prec, rec, _ = sklearn.metrics.precision_recall_curve(labels, scores)

    if is_test:
        # plot precision-recall curve for test set
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(rec, prec, color='b', alpha=0.2,
                 where='post')

        plt.fill_between(rec, prec, alpha=0.2, color='b', **step_kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))

        plt.savefig('prec_rec_curve')

    return acc, precision_score, recall_score, f1, average_precision


def norm_vecs_by_row(x):
    """from Gluon docs"""
    return x / mx.nd.sqrt(mx.nd.sum(x * x, axis=1) + 1E-10).reshape((-1, 1))


def get_knn(vocab, k, word):
    """from Gluon docs"""

    word_vec = vocab.embedding[word].reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(vocab.embedding.idx_to_vec)
    dot_prod = mx.nd.dot(vocab_vecs, word_vec)
    indices = mx.nd.topk(dot_prod.reshape((len(vocab), )), k=k+1, ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    # Remove unknown and input tokens.
    return vocab.to_tokens(indices[1:])
    

if __name__ == '__main__':

    # load the vocab and datasets (train, val, test)
    train_file = args.train_file
    val_file = args.val_file
    test_file = args.test_file

    vocab, train_dataset, val_dataset, test_dataset = load_data.load_dataset(train_file, val_file, test_file)

    ## get the pre-trained word embedding
    glove_twitter = nlp.embedding.create('glove', source=args.embedding_source)
    vocab.set_embedding(glove_twitter)
    shape = vocab.embedding['hello'].shape[0]
    # set unk token to random embedding weights
    vocab.embedding['<unk>'] = mx.nd.random.randn(shape, loc=0, scale=1)

    ctx = mx.cpu() ## or mx.gpu(N) if GPU device N is available
    
    train_classifier(vocab, train_dataset, val_dataset, test_dataset, ctx)
