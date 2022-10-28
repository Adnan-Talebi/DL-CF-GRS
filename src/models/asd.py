def mlp_agg_dense(model, k, dataset, seed):
    ds_shape, nuser, nitems, ds_code = dataset.get_shape(), dataset.get_num_users(), dataset.get_num_items(), dataset.get_data_code()
    input_layer = layers.Input(shape=ds_shape, name="entrada")
    regs=[0,0]
    
    e_user = layers.Dense(
        k,
        name=AGG_PREFIX+"emb_u",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[0])
    )(layers.Lambda(lambda x: x[:, 0:nuser])(input_layer))

    user_latent = Flatten()(e_user)

    vector = user_latent
    # MLP layers
    for idx in [64,32,16,8]:
        layer = Dense(idx, kernel_regularizer= l2(0), activation='relu', name = AGG_PREFIX+'layer%d' %idx)
        vector = layer(vector)
    
    # Connect to dense instead of go wide to narrow again
    # multihot_user = Dense(nuser, kernel_regularizer= l2(0), activation='relu', name='agg_as_MLP')(vector)
    #mlp_agg = Concatenate()([group_embedding, one_hot_item])
    group_embedding = vector
    
    model.trainable = False

    """
    user_input
    emb_u
    emb_u_flat
    item_input
    emb_i
    emb_i_flat
    concatenacion
    layerX
    prediction
    """
    
    one_hot_item = layers.Lambda(lambda x: x[:, nuser:])(input_layer)
    item_x = model.get_layer('emb_i')(one_hot_item)
    item_x = model.get_layer('emb_i_flat')(item_x)
    mlp_agg = Concatenate()([group_embedding, item_x])
    
    for idx in [64,32,16,8]:
        mlp_agg = model.get_layer('layer%d'%idx)(mlp_agg)
    
    output_layer = model.get_layer('prediction')(mlp_agg)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer, name=AGG_PREFIX+model.name)
    model.summary()
    
    return model