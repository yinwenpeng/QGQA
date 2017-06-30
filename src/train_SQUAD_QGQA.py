import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time
from os import system
import numpy as np
import theano
import theano.tensor as T
import codecs
import random
import nltk

from cis.deep.utils.theano import debug_print

from load_SQUAD import load_QGQA, load_glove, refine_decoder_predictions, extract_ansList_attentionList, extract_ansList_attentionList_maxlen5, MacroF1, load_word2vec, load_word2vec_to_init
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import Bd_LSTM_Batch_Tensor_Input_with_Mask_Concate,LSTM_Decoder_Test_with_Mask, LSTM_Decoder_Train_with_Attention, LSTM_Decoder_Test_with_Attention,store_model_to_file, create_LSTM_para, Bd_LSTM_Batch_Tensor_Input_with_Mask, LSTM_Decoder_Train_with_Mask, create_ensemble_para, create_GRU_para, normalize_matrix, create_conv_para, Matrix_Bit_Shift, Conv_with_input_para, L2norm_paraList
from random import shuffle




#need to try
'''
1) dropout
2) combine google and ai2
'''

def evaluate_lenet5(learning_rate=0.01, n_epochs=2000, batch_size=300, test_batch_size=10000, emb_size=50, hidden_size=50,
                    L2_weight=0.0001, para_len_limit=70, q_len_limit=20, pred_q_len_limit=50, top_n_Qwords=1):

    model_options = locals().copy()
    print "model options", model_options
    rootPath='/mounts/data/proj/wenpeng/Dataset/SQuAD/';
    rng = np.random.RandomState(23455)


    word2id = {}
    train_para_list, train_para_mask, train_Q_list, train_Q_mask, train_start_list,train_end_list, _, word2id=load_QGQA(word2id, para_len_limit,q_len_limit, top_n_Qwords, True)
    train_size=len(train_para_list)
    if train_size!=len(train_Q_list) or train_size!=len(train_start_list) or train_size!=len(train_para_mask):
        print 'train_size!=len(Q_list) or train_size!=len(label_list) or train_size!=len(para_mask)'
        exit(0)

    test_para_list, test_para_mask, test_Q_list, test_Q_mask, test_start_list,test_end_list, _, word2id=load_QGQA(word2id, para_len_limit,q_len_limit, top_n_Qwords, False)
    test_size =len(test_para_list)

    train_para_list = np.asarray(train_para_list, dtype='int32')
    train_para_mask = np.asarray(train_para_mask, dtype=theano.config.floatX)

    train_Q_list = np.asarray(train_Q_list, dtype='int32')
    train_Q_mask = np.asarray(train_Q_mask, dtype=theano.config.floatX)

    train_start_list = np.asarray(train_start_list, dtype='int32')
    train_end_list = np.asarray(train_end_list, dtype='int32')

    test_para_list = np.asarray(test_para_list, dtype='int32')
    test_para_mask = np.asarray(test_para_mask, dtype=theano.config.floatX)

    test_Q_list = np.asarray(test_Q_list, dtype='int32')
    test_Q_mask = np.asarray(test_Q_mask, dtype=theano.config.floatX)

    test_start_list = np.asarray(test_start_list, dtype='int32')
    test_end_list = np.asarray(test_end_list, dtype='int32')

    vocab_size = len(word2id)+1

#     shared_decoder_mask = [0]*vocab_size
#     shared_decoder_mask[0]=1#we need this pad token in generated text
#     for id in train_top_Q_wordids:
#         shared_decoder_mask[id]=1
#     shared_decoder_mask=theano.shared(value=np.asarray(shared_decoder_mask, dtype='int32'), borrow=True)  #



    rand_values=random_value_normal((vocab_size, emb_size), theano.config.floatX, np.random.RandomState(1234))
    rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    id2word = {y:x for x,y in word2id.iteritems()}
    word2vec=load_glove()
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=rand_values, borrow=True)
    
    train_top_Q_wordids=set()
    wh_words=['What','Which','Where', 'When','Who', 'Whom','Whose', 'Why', 'How', 'far', 'many', 'much', 'long']
    for word in wh_words:
        idd = word2id.get(word)
        if idd is not None:
            train_top_Q_wordids.add(idd)
        iddd = word2id.get(word.lower())
        if iddd is not None:
            train_top_Q_wordids.add(iddd)


    paragraph = T.imatrix('paragraph')
    questions_encoderIDs = T.imatrix() # is ground truth,
    questions_decoderIDS = T.imatrix() #note we convert then from encoder vocab id to decoder vocab id
    decoder_vocab = T.ivector()
    start_indices= T.ivector() #batch
    end_indices = T.ivector() #batch
    para_mask=T.fmatrix('para_mask')
    q_mask=T.fmatrix('q_mask')



    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    true_batch_size=paragraph.shape[0]
    paragraph_input = embeddings[paragraph.flatten()].reshape((true_batch_size, para_len_limit, emb_size)).dimshuffle(0, 2,1) #(batch, emb_size, para_len)
    q_input = embeddings[questions_encoderIDs.flatten()].reshape((true_batch_size, q_len_limit, emb_size)).dimshuffle(0, 2,1)
    decoder_vocab_embs = embeddings[decoder_vocab]


    fwd_LSTM_para_dict=create_LSTM_para(rng, emb_size, hidden_size)
    bwd_LSTM_para_dict=create_LSTM_para(rng, emb_size, hidden_size)
    paragraph_para=fwd_LSTM_para_dict.values()+ bwd_LSTM_para_dict.values()# .values returns a list of parameters
    paragraph_model=Bd_LSTM_Batch_Tensor_Input_with_Mask_Concate(paragraph_input, para_mask,  hidden_size, fwd_LSTM_para_dict, bwd_LSTM_para_dict)
    paragraph_reps_tensor3=paragraph_model.output_tensor #(batch, 2*hidden, paralen)

    batch_ids=T.arange(true_batch_size)
    ans_heads=paragraph_reps_tensor3[batch_ids,:,start_indices]
    ans_tails=paragraph_reps_tensor3[batch_ids,:,end_indices]

    l_context_heads = paragraph_reps_tensor3[:,:,0]
    l_context_tails = paragraph_reps_tensor3[batch_ids,:,start_indices-1]

    r_context_heads = paragraph_reps_tensor3[batch_ids,:,end_indices+1]
    r_context_tails = paragraph_reps_tensor3[:,:,-1]

    encoder_reps = T.concatenate([l_context_heads,l_context_tails, ans_heads, ans_tails, r_context_heads, r_context_tails], axis=1) #(batch, 6*2hidden_size)


    decoder_para_dict=create_LSTM_para(rng, emb_size+12*hidden_size, emb_size)

    attention_para_dict1=create_LSTM_para(rng, 2*hidden_size, hidden_size)
    attention_para_dict2=create_LSTM_para(rng, 2*hidden_size, hidden_size)



    '''
    train
    '''
    groundtruth_as_input = T.concatenate([T.alloc(np.asarray(0., dtype=theano.config.floatX),true_batch_size,emb_size,1), q_input[:,:,:-1]], axis=2)
#     decoder =  LSTM_Decoder_Train_with_Mask(groundtruth_as_input, encoder_reps, decoder_vocab_embs, q_mask, emb_size, decoder_para_dict)
#X, Encoder_Tensor_Rep, Encoder_Mask, start_indices, end_indices, vocab_embs, Mask, emb_size, hidden_size, tparams, attention_para_dict1, attention_para_dict2
    decoder = LSTM_Decoder_Train_with_Attention(groundtruth_as_input, paragraph_reps_tensor3, para_mask, start_indices, end_indices, decoder_vocab_embs,q_mask,emb_size,hidden_size,decoder_para_dict,attention_para_dict1, attention_para_dict2)

    prob_matrix = decoder.prob_matrix  #(batch*senlen, decoder_vocab_size)
    probs = prob_matrix[T.arange(true_batch_size*q_len_limit),questions_decoderIDS.flatten()]
    mask_probs = probs[(q_mask.flatten()).nonzero()]
    #we shift question word ids so that in current step, the prob of previsouly predicted id gets lower and lower
    shifted_question_ids = T.concatenate([T.alloc(np.asarray(0, dtype='int32'),true_batch_size,1), questions_decoderIDS[:,:-1]], axis=1)
    probs_to_minimize = prob_matrix[T.arange(true_batch_size*q_len_limit),shifted_question_ids.flatten()]
    mask_probs_to_minimize = probs_to_minimize[(q_mask.flatten()).nonzero()]


    #loss train

    loss=-T.mean(T.log(mask_probs))+T.mean(T.exp(mask_probs_to_minimize))
    cost=loss#+ConvGRU_1.error#
    params = [embeddings]+paragraph_para+decoder_para_dict.values()+attention_para_dict1.values()+attention_para_dict2.values()

    accumulator=[]
    for para_i in params:
        eps_p=np.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
#         print grad_i.type
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-8)))   #AdaGrad
        updates.append((acc_i, acc))

#     #test decoder mask
#     raw_masks = T.zeros((true_batch_size, vocab_size), dtype='int32')
#     x_axis = T.repeat(T.arange(true_batch_size).dimshuffle(0,'x'), paragraph.shape[1], axis=1)
#     input_specific_masks = T.set_subtensor(raw_masks[x_axis.flatten(),paragraph.flatten()],1)
#     overall_test_decoder_mask = T.or_(input_specific_masks, shared_decoder_mask.dimshuffle('x',0))  #(batch, vocab_size)
#     overall_test_decoder_mask=(1.0-overall_test_decoder_mask)*(overall_test_decoder_mask-10)

    '''
    testing
    '''
#     test_decoder =  LSTM_Decoder_Test_with_Mask(q_len_limit, encoder_reps, decoder_vocab_embs, emb_size, decoder_para_dict)
#nsteps, Encoder_Tensor_Rep, Encoder_Mask, start_indices, end_indices, vocab_embs, emb_size,hidden_size, tparams,attention_para_dict1, attention_para_dict2
    test_decoder = LSTM_Decoder_Test_with_Attention(pred_q_len_limit,paragraph_reps_tensor3, para_mask, start_indices, end_indices, decoder_vocab_embs, emb_size,hidden_size,decoder_para_dict,attention_para_dict1, attention_para_dict2)
    predictions = test_decoder.output_id_matrix #(batch, q_len_limit)


    train_model = theano.function([paragraph, questions_encoderIDs,questions_decoderIDS, decoder_vocab, start_indices, end_indices,para_mask, q_mask], cost, updates=updates,on_unused_input='ignore')

    test_model = theano.function([paragraph, decoder_vocab, start_indices, end_indices,para_mask], predictions, on_unused_input='ignore')




    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 500000000000000  # look as this many examples regardless


    best_params = None
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.time()
    mid_time = start_time
    past_time= mid_time
    epoch = 0
    done_looping = False


    #para_list, Q_list, label_list, mask, vocab_size=load_train()
    n_train_batches=train_size/batch_size
#     remain_train=train_size%batch_size
    train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size]


    n_test_batches=test_size/test_batch_size
    remain_test=test_size%test_batch_size
    test_batch_start=list(np.arange(n_test_batches)*test_batch_size)+[test_size-remain_test]


    max_bleuscore=0.0
    max_exact_acc=0.0
    cost_i=0.0
    train_ids = range(train_size)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        random.shuffle(train_ids)
        iter_accu=0
        for para_id in train_batch_start:
            # iter means how many batches have been runed, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            sub_Qs = train_Q_list[para_id:para_id+batch_size]
            decoder_vocab_set = train_top_Q_wordids | set(list(np.unique(sub_Qs)))
            decoder_vocab_batch = sorted(decoder_vocab_set) # a list of ids in order
            map_encoderid2decoderid={}
            for encoderID in decoder_vocab_set:
                decoderID= decoder_vocab_batch.index(encoderID)
                map_encoderid2decoderid[encoderID] = decoderID
            Decoder_train_Q_list = []
            for id in sub_Qs.flatten():
                Decoder_train_Q_list.append(map_encoderid2decoderid.get(id))
            Decoder_train_Q_list = np.asarray(Decoder_train_Q_list, dtype='int32').reshape((batch_size, sub_Qs.shape[1]))
            decoder_vocab_batch = np.asarray(decoder_vocab_batch, dtype='int32')

            cost_i+= train_model(
                                train_para_list[para_id:para_id+batch_size],
                                train_Q_list[para_id:para_id+batch_size],
                                Decoder_train_Q_list,
                                decoder_vocab_batch,
                                train_start_list[para_id:para_id+batch_size],
                                train_end_list[para_id:para_id+batch_size],
                                train_para_mask[para_id:para_id+batch_size],
                                train_Q_mask[para_id:para_id+batch_size])

            #print iter
            if iter%100==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
#                 print 'Testing...'
                past_time = time.time()
                outputfile=codecs.open('output.txt', 'w', 'utf-8')
                referencefile = codecs.open('reference.txt', 'w', 'utf-8')

                bleu_scores = []
                for idd, test_para_id in enumerate(test_batch_start):
                    sub_Qs = test_Q_list[test_para_id:test_para_id+test_batch_size]
                    decoder_vocab_set = train_top_Q_wordids | set(list(np.unique(sub_Qs)))
                    decoder_vocab_batch = sorted(decoder_vocab_set) # a list of ids in order

                    map_decoderid2encoderid={}
                    for encoderID in decoder_vocab_set:
                        decoderID= decoder_vocab_batch.index(encoderID)
                        map_decoderid2encoderid[decoderID] = encoderID
#                     Decoder_train_Q_list = []
#                     for id in sub_Qs.flatten():
#                         Decoder_train_Q_list.append(map_encoderid2decoderid.get(id))
#                     Decoder_train_Q_list = np.asarray(Decoder_train_Q_list, dtype='int32').reshape((batch_size, sub_Qs.shape[1]))
                    decoder_vocab_batch = np.asarray(decoder_vocab_batch, dtype='int32')
                    pred_id_in_batch = test_model(
                                        test_para_list[test_para_id:test_para_id+test_batch_size],
                                        decoder_vocab_batch,
                                        test_start_list[test_para_id:test_para_id+test_batch_size],
                                        test_end_list[test_para_id:test_para_id+test_batch_size],
                                        test_para_mask[test_para_id:test_para_id+test_batch_size])  #(batch, senlen)
                    ground_truths = sub_Qs
                    ground_mask = test_Q_mask[test_para_id:test_para_id+test_batch_size]

                    back_pred_id_in_batch=[map_decoderid2encoderid.get(id) for id in pred_id_in_batch.flatten()]


                    if idd == len(test_batch_start)-1:
                        true_test_batch_size = remain_test
                    else:
                        true_test_batch_size=test_batch_size
                    for i in range(true_test_batch_size):
#                         print 'pred_id_in_batch[i]:', pred_id_in_batch[i]
                        refined_preds, refined_g = refine_decoder_predictions(back_pred_id_in_batch[i*pred_q_len_limit:(i+1)*pred_q_len_limit], ground_truths[i], ground_mask[i])
#                         bleu_i = nltk.translate.bleu_score.sentence_bleu([refined_g], refined_preds)
#                         bleu_scores.append(bleu_i)
                        outputfile.write(' '.join([id2word.get(id) for id in refined_preds])+'\n')
                        referencefile.write(' '.join([id2word.get(id) for id in refined_g])+'\n')

#                 bleuscore =  np.average(np.array(bleu_scores))
                outputfile.close()
                referencefile.close()
                system('perl multi-bleu.perl reference.txt < output.txt')
#                 if max_bleuscore < bleuscore:
#                     max_bleuscore = bleuscore
#                 print '\t\t\t\t\t\t current bleu: ', bleuscore, ' ; max bleu:', max_bleuscore





            if patience <= iter:
                done_looping = True
                break

        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()

        #print 'Batch_size: ', update_freq
    end_time = time.time()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))







if __name__ == '__main__':
    evaluate_lenet5()
