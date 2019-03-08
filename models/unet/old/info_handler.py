"""
Taken from / credit to:
https://github.com/tensorspace/fashion_mnist_and_MI_visualization
"""

import torch, pickle
import training_metadata as tm
import numpy as np

#import keras.backend as K

class InfoHandler:

    def __init__(self, model, params, trn):

        self.save_to_dir = tm.construct_file().replace(".txt", "info_log/")
        
        self.model = model
        
        self.params = params

        # The (x, y) training data
        self.trn = trn

    def save_info_func(self, epoch):
        # Only log activity for some epochs.  Mainly this is to make things run faster.
        
        return True # save all fr now...
        
        """
        if epoch < 20:       # Log for all first 20 epochs
            return True
        elif epoch < 100:    # Then for every 5th epoch
            return (epoch % 5 == 0)
        elif epoch < 200:    # Then every 10th
            return (epoch % 10 == 0)
        else:                # Then every 100th
            return (epoch % 100 == 0)
        """

    def on_train_begin(self, logs={}):
        """
        When training begins
        """

        # Indices we will keep track of
        self.layerixs = []

        # Functions return weights of each layer
        self.layerweights = []

        # Functions return activity of each layer
        self.layerfuncs = []

        for lndx, l in enumerate(self.model.layers):
            if hasattr(l, 'weight'):

                self.layerixs.append(lndx)

                self.layerweights.append(l.weight)
                
                # TODO - IMPLEMENT in pytorch
                self.layerfuncs.append(K.function(self.model.inputs, [l.output,])) # original

                """
                # Some explanation of what K.function does: 
                
                final_conv_layer = get_output_layer(model, "conv5_3")
            
                get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
                # E.g. 
                #   Takes input of shape model.layers[0] (placeholder)
                #   Returns the output of the final conv layer, and the prediction
                
                [conv_outputs, predictions] = get_output([img])
                """

        # TODO - IMPLEMENT in pytorch (what are inputs, sample_weights, targets?)
        input_tensors = [self.model.inputs[0],
                         self.model.sample_weights[0],
                         self.model.targets[0],
                         K.learning_phase()] ## 0 or 1 (false, true)
        
        # TODO - implement in pytorch
        # Get gradients of all the relevant layers at once
        grads = self.model.optimizer.get_gradients(self.model.total_loss, self.layerweights)
        
        # TODO - implement in pytorch
        self.get_gradients = K.function(inputs=input_tensors, outputs=grads)
        
        # TODO - implement in pytorch
        # Get cross-entropy loss
        self.get_loss = K.function(inputs=input_tensors, outputs=[self.model.total_loss,])
            
    def on_epoch_begin(self, epoch, logs={}):
        if self.save_info_func is not None and not self.save_info_func(epoch):
            # Don't log this epoch
            self._log_gradients = False
        else:
            # We will log this epoch.
            # For each batch in this epoch, we will save the gradients (in on_batch_begin)
            # We will then compute means and vars of these gradients
            
            self._log_gradients = True
            self._batch_weightnorm = []
            
            # Gradients of each batch - no grad for 1st one 
            self._batch_gradients = [ [] for _ in self.model.layers[1:] ]
            
            # Indexes of all the training data samples. These are shuffled and read-in in chunks of SGD_BATCHSIZE
            ixs = list(range(len(self.trn['X'])))
            np.random.shuffle(ixs)
            self._batch_todo_ixs = ixs

    def on_batch_begin(self, batch, logs={}):
        """At start of batch, track info."""

        # We are not keeping track of batch gradients, so do nothing
        if not self._log_gradients:
            return
        
        # Sample a batch
        batchsize = batch['inputs'].size(0)
        cur_ixs = self._batch_todo_ixs[:batchsize]
        
        # Advance the indexing, so next on_batch_begin samples a different batch
        self._batch_todo_ixs = self._batch_todo_ixs[batchsize:]
        
        # Get gradients for this batch

        # ORIGINAL (TODO - fix functional and self.get_gradients would be implemented)
        inputs = [batch['inputs'],  # Inputs
                  [1,]*len(cur_ixs),      # Uniform sample weights
                  batch['labels'],  # Outputs
                  1                       # Training phase
                 ]
        for lndx, g in enumerate(self.get_gradients(inputs)):
            # g is gradients for weights of lndx's layer
            oneDgrad = g.view((-1, 1))                  # Flatten to one dimensional vector
            self._batch_gradients[lndx].append(oneDgrad)        

        """
        ## ATTEMPT TO REPLACE ##

        for lndx in self.layerixs:
            
            if lndx == 0:
                # Won't get a gradient for 1st one obvs
                continue
            
            l = self.model.layers[lndx]
            
            # g is gradients for weights of lndx's layer
            try:
                g = l.weight.grad
                print("layer", lndx, l, "grad", g)
            except:
                raise Exception("failed on layer", lndx, l)

            oneDgrad = g.view((-1, 1))                  # Flatten to one dimensional vector
            self._batch_gradients[lndx].append(oneDgrad)
        ######################
        """

    def on_epoch_end(self, epoch, logs={}):
        """At end of epoch, track what needs to be tracked."""

        def get_activation(lndx):
            # Gets the activation of the indexed layer
            def hook(model, input, output):
                self.layerfuncs[lndx] = output.detach()
            return hook

        if self.save_info_func is not None and not self.save_info_func(epoch):
            # Don't log this epoch
            return
        
        # Get overall performance
        loss = {}
        for cdata, cdataname, istrain in ((self.trn,'trn',1), (self.tst, 'tst',0)):
            loss[cdataname] = self.get_loss([cdata['X'], [1,]*len(cdata['X']), cdata['Y'], istrain])[0].flat[0]
            
        data = {
            'weights_norm' : [],   # L2 norm of weights
            'gradmean'     : [],   # Mean of gradients
            'gradstd'      : [],   # Std of gradients
            'activity_tst' : []    # Activity in each layer for test set
        }
        
        for lndx, layerix in enumerate(self.layerixs):
            clayer = self.model.layers[layerix]
            
            data['weights_norm'].append( np.linalg.norm(clayer.weight.cpu().numpy()) )
            
            stackedgrads = np.stack(self._batch_gradients[lndx], axis=1)
            data['gradmean'    ].append( np.linalg.norm(stackedgrads.mean(axis=1)) )
            data['gradstd'     ].append( np.linalg.norm(stackedgrads.std(axis=1)) )

            # ORIGINAL (changed X format) - TODO: fix layerfuncs
            data['activity_tst'].append(self.layerfuncs[lndx]([self.tst['X'],])[0])

            """
            ## ATTEMPT TO REPLACE ORIGINAL ##
            
            # TODO - just calculate the activation here - the functions don't work
            self.model.layers[layerix].register_forward_hook(get_activation(layerix))
            
            # x = torch.randn(1, 25) # not sure what input should be
            # output = model(x) # SHOULD already be full - may be needed 

            print("activation", self.layerfuncs[layerix].shape)
            
            #TODO - check layerix (actual layer index value) not lndx (layer index within list)
            data['activity_tst'].append(self.layerfuncs[layerix]) 

            ########################
            """

        fname = self.save_to_dir + "epoch%08d" % epoch
        print("Saving", fname)
        with open(fname, 'wb') as f:
             pickle.dump({'ACTIVATION':self.cfg['ACTIVATION'], 'epoch':epoch, 'data':data, 'loss':loss}, f, pickle.HIGHEST_PROTOCOL)
