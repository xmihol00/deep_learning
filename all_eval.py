
def eval_models(models, models_to_train, x_train, y_train):
    FC_SP_16_256.load_weights("./models/FC_SP_16_256/FC_SP_16_256").expect_partial()
    FC_SP_16_256_pred = FC_SP_16_256.predict(x_test)
    FC_SP_16_256_accuracy = (np.argmax(FC_SP_16_256_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_SP_16_256 accuracy:  {FC_SP_16_256_accuracy * 100:.2f} %")
    del FC_SP_16_256
    
    
    FC_MP_16_256.load_weights("./models/FC_MP_16_256/FC_MP_16_256").expect_partial()
    FC_MP_16_256_pred = FC_MP_16_256.predict(x_test)
    FC_MP_16_256_accuracy = (np.argmax(FC_MP_16_256_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_16_256 accuracy:  {FC_MP_16_256_accuracy * 100:.2f} %")
    del FC_MP_16_256
    
    
    FC_MP_32_512.load_weights("./models/FC_MP_32_512/FC_MP_32_512").expect_partial()
    FC_MP_32_512_pred = FC_MP_32_512.predict(x_test)
    FC_MP_32_512_accuracy = (np.argmax(FC_MP_32_512_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_32_512 accuracy:  {FC_MP_32_512_accuracy * 100:.2f} %")
    del FC_MP_32_512
    
    
    VGG_2B_32_64.load_weights("./models/VGG_2B_32_64/VGG_2B_32_64").expect_partial()
    VGG_2B_32_64_pred = VGG_2B_32_64.predict(x_test)
    VGG_2B_32_64_accuracy = (np.argmax(VGG_2B_32_64_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_2B_32_64 accuracy:  {VGG_2B_32_64_accuracy * 100:.2f} %")
    del VGG_2B_32_64
    
    
    VGG_3B_16_64.load_weights("./models/VGG_3B_16_64/VGG_3B_16_64").expect_partial()
    VGG_3B_16_64_pred = VGG_3B_16_64.predict(x_test)
    VGG_3B_16_64_accuracy = (np.argmax(VGG_3B_16_64_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_16_64 accuracy:  {VGG_3B_16_64_accuracy * 100:.2f} %")
    del VGG_3B_16_64
    
    
    VGG_3B_32_128.load_weights("./models/VGG_3B_32_128/VGG_3B_32_128").expect_partial()
    VGG_3B_32_128_pred = VGG_3B_32_128.predict(x_test)
    VGG_3B_32_128_accuracy = (np.argmax(VGG_3B_32_128_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_32_128 accuracy: {VGG_3B_32_128_accuracy * 100:.2f} %")
    del VGG_3B_32_128
    
    
    VGG_2B_32_64_dropout_03.load_weights("./models/VGG_2B_32_64/VGG_2B_32_64_dropout_03").expect_partial()
    VGG_2B_32_64_dropout_03_pred = VGG_2B_32_64_dropout_03.predict(x_test)
    VGG_2B_32_64_dropout_03_accuracy = (np.argmax(VGG_2B_32_64_dropout_03_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_2B_32_64_dropout_03 accuracy:  {VGG_2B_32_64_dropout_03_accuracy * 100:.2f} %")
    del VGG_2B_32_64_dropout_03
    
    
    VGG_3B_16_64_dropout_03.load_weights("./models/VGG_3B_16_64/VGG_3B_16_64_dropout_03").expect_partial()
    VGG_3B_16_64_dropout_03_pred = VGG_3B_16_64_dropout_03.predict(x_test)
    VGG_3B_16_64_dropout_03_accuracy = (np.argmax(VGG_3B_16_64_dropout_03_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_16_64_dropout_03 accuracy:  {VGG_3B_16_64_dropout_03_accuracy * 100:.2f} %")
    del VGG_3B_16_64_dropout_03
    
    
    VGG_3B_32_128_dropout_03.load_weights("./models/VGG_3B_32_128/VGG_3B_32_128_dropout_03").expect_partial()
    VGG_3B_32_128_dropout_03_pred = VGG_3B_32_128_dropout_03.predict(x_test)
    VGG_3B_32_128_dropout_03_accuracy = (np.argmax(VGG_3B_32_128_dropout_03_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_32_128_dropout_03 accuracy: {VGG_3B_32_128_dropout_03_accuracy * 100:.2f} %")
    del VGG_3B_32_128_dropout_03
    
    
    
    VGG_2B_32_64_dropout_04.load_weights("./models/VGG_2B_32_64/VGG_2B_32_64_dropout_04").expect_partial()
    VGG_2B_32_64_dropout_04_pred = VGG_2B_32_64_dropout_04.predict(x_test)
    VGG_2B_32_64_dropout_04_accuracy = (np.argmax(VGG_2B_32_64_dropout_04_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_2B_32_64_dropout_04 accuracy:  {VGG_2B_32_64_dropout_04_accuracy * 100:.2f} %")
    del VGG_2B_32_64_dropout_04
    
    
    VGG_3B_16_64_dropout_04.load_weights("./models/VGG_3B_16_64/VGG_3B_16_64_dropout_04").expect_partial()
    VGG_3B_16_64_dropout_04_pred = VGG_3B_16_64_dropout_04.predict(x_test)
    VGG_3B_16_64_dropout_04_accuracy = (np.argmax(VGG_3B_16_64_dropout_04_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_16_64_dropout_04 accuracy:  {VGG_3B_16_64_dropout_04_accuracy * 100:.2f} %")
    del VGG_3B_16_64_dropout_04
    
    
    VGG_3B_32_128_dropout_04.load_weights("./models/VGG_3B_32_128/VGG_3B_32_128_dropout_04").expect_partial()
    VGG_3B_32_128_dropout_04_pred = VGG_3B_32_128_dropout_04.predict(x_test)
    VGG_3B_32_128_dropout_04_accuracy = (np.argmax(VGG_3B_32_128_dropout_04_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_32_128_dropout_04 accuracy: {VGG_3B_32_128_dropout_04_accuracy * 100:.2f} %")
    del VGG_3B_32_128_dropout_04
    
    
    
    FC_SP_16_256_batch_norm.load_weights("./models/FC_SP_16_256/FC_SP_16_256_batch_norm").expect_partial()
    FC_SP_16_256_batch_norm_pred = FC_SP_16_256_batch_norm.predict(x_test)
    FC_SP_16_256_batch_norm_accuracy = (np.argmax(FC_SP_16_256_batch_norm_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_SP_16_256_batch_norm accuracy:  {FC_SP_16_256_batch_norm_accuracy * 100:.2f} %")
    del FC_SP_16_256_batch_norm
    

    FC_MP_16_256_batch_norm.load_weights("./models/FC_MP_16_256/FC_MP_16_256_batch_norm").expect_partial()
    FC_MP_16_256_batch_norm_pred = FC_MP_16_256_batch_norm.predict(x_test)
    FC_MP_16_256_batch_norm_accuracy = (np.argmax(FC_MP_16_256_batch_norm_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_16_256_batch_norm accuracy:  {FC_MP_16_256_batch_norm_accuracy * 100:.2f} %")
    del FC_MP_16_256_batch_norm


    FC_MP_32_512_batch_norm.load_weights("./models/FC_MP_32_512/FC_MP_32_512_batch_norm").expect_partial()
    FC_MP_32_512_batch_norm_pred = FC_MP_32_512_batch_norm.predict(x_test)
    FC_MP_32_512_batch_norm_accuracy = (np.argmax(FC_MP_32_512_batch_norm_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_32_512_batch_norm accuracy:  {FC_MP_32_512_batch_norm_accuracy * 100:.2f} %")
    del FC_MP_32_512_batch_norm


    VGG_2B_32_64_dropout_05.load_weights("./models/VGG_2B_32_64/VGG_2B_32_64_dropout_05").expect_partial()
    VGG_2B_32_64_dropout_05_pred = VGG_2B_32_64_dropout_05.predict(x_test)
    VGG_2B_32_64_dropout_05_accuracy = (np.argmax(VGG_2B_32_64_dropout_05_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_2B_32_64_dropout_05 accuracy:  {VGG_2B_32_64_dropout_05_accuracy * 100:.2f} %")
    del VGG_2B_32_64_dropout_05


    VGG_3B_16_64_dropout_05.load_weights("./models/VGG_3B_16_64/VGG_3B_16_64_dropout_05").expect_partial()
    VGG_3B_16_64_dropout_05_pred = VGG_3B_16_64_dropout_05.predict(x_test)
    VGG_3B_16_64_dropout_05_accuracy = (np.argmax(VGG_3B_16_64_dropout_05_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_16_64_dropout_05 accuracy:  {VGG_3B_16_64_dropout_05_accuracy * 100:.2f} %")
    del VGG_3B_16_64_dropout_05


    VGG_3B_32_128_dropout_05.load_weights("./models/VGG_3B_32_128/VGG_3B_32_128_dropout_05").expect_partial()
    VGG_3B_32_128_dropout_05_pred = VGG_3B_32_128_dropout_05.predict(x_test)
    VGG_3B_32_128_dropout_05_accuracy = (np.argmax(VGG_3B_32_128_dropout_05_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_32_128_dropout_05 accuracy: {VGG_3B_32_128_dropout_05_accuracy * 100:.2f} %")
    del VGG_3B_32_128_dropout_05



    FC_SP_16_256_l1_00001.load_weights("./models/FC_SP_16_256/FC_SP_16_256_l1_00001").expect_partial()
    FC_SP_16_256_l1_00001_pred = FC_SP_16_256_l1_00001.predict(x_test)
    FC_SP_16_256_l1_00001_accuracy = (np.argmax(FC_SP_16_256_l1_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_SP_16_256_l1_00001 accuracy:  {FC_SP_16_256_l1_00001_accuracy * 100:.2f} %")
    del FC_SP_16_256_l1_00001


    FC_MP_16_256_l1_00001.load_weights("./models/FC_MP_16_256/FC_MP_16_256_l1_00001").expect_partial()
    FC_MP_16_256_l1_00001_pred = FC_MP_16_256_l1_00001.predict(x_test)
    FC_MP_16_256_l1_00001_accuracy = (np.argmax(FC_MP_16_256_l1_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_16_256_l1_00001 accuracy:  {FC_MP_16_256_l1_00001_accuracy * 100:.2f} %")
    del FC_MP_16_256_l1_00001


    FC_MP_32_512_l1_00001.load_weights("./models/FC_MP_32_512/FC_MP_32_512_l1_00001").expect_partial()
    FC_MP_32_512_l1_00001_pred = FC_MP_32_512_l1_00001.predict(x_test)
    FC_MP_32_512_l1_00001_accuracy = (np.argmax(FC_MP_32_512_l1_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_32_512_l1_00001 accuracy:  {FC_MP_32_512_l1_00001_accuracy * 100:.2f} %")
    del FC_MP_32_512_l1_00001


    VGG_2B_32_64_l1_00001.load_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1_00001").expect_partial()
    VGG_2B_32_64_l1_00001_pred = VGG_2B_32_64_l1_00001.predict(x_test)
    VGG_2B_32_64_l1_00001_accuracy = (np.argmax(VGG_2B_32_64_l1_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_2B_32_64_l1_00001 accuracy:  {VGG_2B_32_64_l1_00001_accuracy * 100:.2f} %")
    del VGG_2B_32_64_l1_00001


    VGG_3B_16_64_l1_00001.load_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1_00001").expect_partial()
    VGG_3B_16_64_l1_00001_pred = VGG_3B_16_64_l1_00001.predict(x_test)
    VGG_3B_16_64_l1_00001_accuracy = (np.argmax(VGG_3B_16_64_l1_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_16_64_l1_00001 accuracy:  {VGG_3B_16_64_l1_00001_accuracy * 100:.2f} %")
    del VGG_3B_16_64_l1_00001


    VGG_3B_32_128_l1_00001.load_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1_00001").expect_partial()
    VGG_3B_32_128_l1_00001_pred = VGG_3B_32_128_l1_00001.predict(x_test)
    VGG_3B_32_128_l1_00001_accuracy = (np.argmax(VGG_3B_32_128_l1_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_32_128_l1_00001 accuracy: {VGG_3B_32_128_l1_00001_accuracy * 100:.2f} %")
    del VGG_3B_32_128_l1_00001


    FC_SP_16_256_l1_0001.load_weights("./models/FC_SP_16_256/FC_SP_16_256_l1_0001").expect_partial()
    FC_SP_16_256_l1_0001_pred = FC_SP_16_256_l1_0001.predict(x_test)
    FC_SP_16_256_l1_0001_accuracy = (np.argmax(FC_SP_16_256_l1_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_SP_16_256_l1_0001 accuracy:  {FC_SP_16_256_l1_0001_accuracy * 100:.2f} %")
    del FC_SP_16_256_l1_0001


    FC_MP_16_256_l1_0001.load_weights("./models/FC_MP_16_256/FC_MP_16_256_l1_0001").expect_partial()
    FC_MP_16_256_l1_0001_pred = FC_MP_16_256_l1_0001.predict(x_test)
    FC_MP_16_256_l1_0001_accuracy = (np.argmax(FC_MP_16_256_l1_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_16_256_l1_0001 accuracy:  {FC_MP_16_256_l1_0001_accuracy * 100:.2f} %")
    del FC_MP_16_256_l1_0001


    FC_MP_32_512_l1_0001.load_weights("./models/FC_MP_32_512/FC_MP_32_512_l1_0001").expect_partial()
    FC_MP_32_512_l1_0001_pred = FC_MP_32_512_l1_0001.predict(x_test)
    FC_MP_32_512_l1_0001_accuracy = (np.argmax(FC_MP_32_512_l1_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_32_512_l1_0001 accuracy:  {FC_MP_32_512_l1_0001_accuracy * 100:.2f} %")
    del FC_MP_32_512_l1_0001


    VGG_2B_32_64_l1_0001.load_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1_0001").expect_partial()
    VGG_2B_32_64_l1_0001_pred = VGG_2B_32_64_l1_0001.predict(x_test)
    VGG_2B_32_64_l1_0001_accuracy = (np.argmax(VGG_2B_32_64_l1_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_2B_32_64_l1_0001 accuracy:  {VGG_2B_32_64_l1_0001_accuracy * 100:.2f} %")
    del VGG_2B_32_64_l1_0001


    VGG_3B_16_64_l1_0001.load_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1_0001").expect_partial()
    VGG_3B_16_64_l1_0001_pred = VGG_3B_16_64_l1_0001.predict(x_test)
    VGG_3B_16_64_l1_0001_accuracy = (np.argmax(VGG_3B_16_64_l1_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_16_64_l1_0001 accuracy:  {VGG_3B_16_64_l1_0001_accuracy * 100:.2f} %")
    del VGG_3B_16_64_l1_0001


    VGG_3B_32_128_l1_0001.load_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1_0001").expect_partial()
    VGG_3B_32_128_l1_0001_pred = VGG_3B_32_128_l1_0001.predict(x_test)
    VGG_3B_32_128_l1_0001_accuracy = (np.argmax(VGG_3B_32_128_l1_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_32_128_l1_0001 accuracy: {VGG_3B_32_128_l1_0001_accuracy * 100:.2f} %")
    del VGG_3B_32_128_l1_0001


    FC_SP_16_256_l1_001.load_weights("./models/FC_SP_16_256/FC_SP_16_256_l1_001").expect_partial()
    FC_SP_16_256_l1_001_pred = FC_SP_16_256_l1_001.predict(x_test)
    FC_SP_16_256_l1_001_accuracy = (np.argmax(FC_SP_16_256_l1_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_SP_16_256_l1_001 accuracy:  {FC_SP_16_256_l1_001_accuracy * 100:.2f} %")
    del FC_SP_16_256_l1_001


    FC_MP_16_256_l1_001.load_weights("./models/FC_MP_16_256/FC_MP_16_256_l1_001").expect_partial()
    FC_MP_16_256_l1_001_pred = FC_MP_16_256_l1_001.predict(x_test)
    FC_MP_16_256_l1_001_accuracy = (np.argmax(FC_MP_16_256_l1_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_16_256_l1_001 accuracy:  {FC_MP_16_256_l1_001_accuracy * 100:.2f} %")
    del FC_MP_16_256_l1_001


    FC_MP_32_512_l1_001.load_weights("./models/FC_MP_32_512/FC_MP_32_512_l1_001").expect_partial()
    FC_MP_32_512_l1_001_pred = FC_MP_32_512_l1_001.predict(x_test)
    FC_MP_32_512_l1_001_accuracy = (np.argmax(FC_MP_32_512_l1_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_32_512_l1_001 accuracy:  {FC_MP_32_512_l1_001_accuracy * 100:.2f} %")
    del FC_MP_32_512_l1_001


    VGG_2B_32_64_l1_001.load_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1_001").expect_partial()
    VGG_2B_32_64_l1_001_pred = VGG_2B_32_64_l1_001.predict(x_test)
    VGG_2B_32_64_l1_001_accuracy = (np.argmax(VGG_2B_32_64_l1_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_2B_32_64_l1_001 accuracy:  {VGG_2B_32_64_l1_001_accuracy * 100:.2f} %")
    del VGG_2B_32_64_l1_001


    VGG_3B_16_64_l1_001.load_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1_001").expect_partial()
    VGG_3B_16_64_l1_001_pred = VGG_3B_16_64_l1_001.predict(x_test)
    VGG_3B_16_64_l1_001_accuracy = (np.argmax(VGG_3B_16_64_l1_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_16_64_l1_001 accuracy:  {VGG_3B_16_64_l1_001_accuracy * 100:.2f} %")
    del VGG_3B_16_64_l1_001


    VGG_3B_32_128_l1_001.load_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1_001").expect_partial()
    VGG_3B_32_128_l1_001_pred = VGG_3B_32_128_l1_001.predict(x_test)
    VGG_3B_32_128_l1_001_accuracy = (np.argmax(VGG_3B_32_128_l1_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_32_128_l1_001 accuracy: {VGG_3B_32_128_l1_001_accuracy * 100:.2f} %")
    del VGG_3B_32_128_l1_001




    FC_SP_16_256_l1l2_00001.load_weights("./models/FC_SP_16_256/FC_SP_16_256_l1l2_00001").expect_partial()
    FC_SP_16_256_l1l2_00001_pred = FC_SP_16_256_l1l2_00001.predict(x_test)
    FC_SP_16_256_l1l2_00001_accuracy = (np.argmax(FC_SP_16_256_l1l2_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_SP_16_256_l1l2_00001 accuracy:  {FC_SP_16_256_l1l2_00001_accuracy * 100:.2f} %")
    del FC_SP_16_256_l1l2_00001


    FC_MP_16_256_l1l2_00001.load_weights("./models/FC_MP_16_256/FC_MP_16_256_l1l2_00001").expect_partial()
    FC_MP_16_256_l1l2_00001_pred = FC_MP_16_256_l1l2_00001.predict(x_test)
    FC_MP_16_256_l1l2_00001_accuracy = (np.argmax(FC_MP_16_256_l1l2_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_16_256_l1l2_00001 accuracy:  {FC_MP_16_256_l1l2_00001_accuracy * 100:.2f} %")
    del FC_MP_16_256_l1l2_00001


    FC_MP_32_512_l1l2_00001.load_weights("./models/FC_MP_32_512/FC_MP_32_512_l1l2_00001").expect_partial()
    FC_MP_32_512_l1l2_00001_pred = FC_MP_32_512_l1l2_00001.predict(x_test)
    FC_MP_32_512_l1l2_00001_accuracy = (np.argmax(FC_MP_32_512_l1l2_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_32_512_l1l2_00001 accuracy:  {FC_MP_32_512_l1l2_00001_accuracy * 100:.2f} %")
    del FC_MP_32_512_l1l2_00001


    VGG_2B_32_64_l1l2_00001.load_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1l2_00001").expect_partial()
    VGG_2B_32_64_l1l2_00001_pred = VGG_2B_32_64_l1l2_00001.predict(x_test)
    VGG_2B_32_64_l1l2_00001_accuracy = (np.argmax(VGG_2B_32_64_l1l2_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_2B_32_64_l1l2_00001 accuracy:  {VGG_2B_32_64_l1l2_00001_accuracy * 100:.2f} %")
    del VGG_2B_32_64_l1l2_00001


    VGG_3B_16_64_l1l2_00001.load_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1l2_00001").expect_partial()
    VGG_3B_16_64_l1l2_00001_pred = VGG_3B_16_64_l1l2_00001.predict(x_test)
    VGG_3B_16_64_l1l2_00001_accuracy = (np.argmax(VGG_3B_16_64_l1l2_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_16_64_l1l2_00001 accuracy:  {VGG_3B_16_64_l1l2_00001_accuracy * 100:.2f} %")
    del VGG_3B_16_64_l1l2_00001


    VGG_3B_32_128_l1l2_00001.load_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1l2_00001").expect_partial()
    VGG_3B_32_128_l1l2_00001_pred = VGG_3B_32_128_l1l2_00001.predict(x_test)
    VGG_3B_32_128_l1l2_00001_accuracy = (np.argmax(VGG_3B_32_128_l1l2_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_32_128_l1l2_00001 accuracy: {VGG_3B_32_128_l1l2_00001_accuracy * 100:.2f} %")
    del VGG_3B_32_128_l1l2_00001




    FC_SP_16_256_l1l2_0001.load_weights("./models/FC_SP_16_256/FC_SP_16_256_l1l2_0001").expect_partial()
    FC_SP_16_256_l1l2_0001_pred = FC_SP_16_256_l1l2_0001.predict(x_test)
    FC_SP_16_256_l1l2_0001_accuracy = (np.argmax(FC_SP_16_256_l1l2_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_SP_16_256_l1l2_0001 accuracy:  {FC_SP_16_256_l1l2_0001_accuracy * 100:.2f} %")
    del FC_SP_16_256_l1l2_0001


    FC_MP_16_256_l1l2_0001.load_weights("./models/FC_MP_16_256/FC_MP_16_256_l1l2_0001").expect_partial()
    FC_MP_16_256_l1l2_0001_pred = FC_MP_16_256_l1l2_0001.predict(x_test)
    FC_MP_16_256_l1l2_0001_accuracy = (np.argmax(FC_MP_16_256_l1l2_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_16_256_l1l2_0001 accuracy:  {FC_MP_16_256_l1l2_0001_accuracy * 100:.2f} %")
    del FC_MP_16_256_l1l2_0001


    FC_MP_32_512_l1l2_0001.load_weights("./models/FC_MP_32_512/FC_MP_32_512_l1l2_0001").expect_partial()
    FC_MP_32_512_l1l2_0001_pred = FC_MP_32_512_l1l2_0001.predict(x_test)
    FC_MP_32_512_l1l2_0001_accuracy = (np.argmax(FC_MP_32_512_l1l2_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_32_512_l1l2_0001 accuracy:  {FC_MP_32_512_l1l2_0001_accuracy * 100:.2f} %")
    del FC_MP_32_512_l1l2_0001


    VGG_2B_32_64_l1l2_0001.load_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1l2_0001").expect_partial()
    VGG_2B_32_64_l1l2_0001_pred = VGG_2B_32_64_l1l2_0001.predict(x_test)
    VGG_2B_32_64_l1l2_0001_accuracy = (np.argmax(VGG_2B_32_64_l1l2_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_2B_32_64_l1l2_0001 accuracy:  {VGG_2B_32_64_l1l2_0001_accuracy * 100:.2f} %")
    del VGG_2B_32_64_l1l2_0001


    VGG_3B_16_64_l1l2_0001.load_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1l2_0001").expect_partial()
    VGG_3B_16_64_l1l2_0001_pred = VGG_3B_16_64_l1l2_0001.predict(x_test)
    VGG_3B_16_64_l1l2_0001_accuracy = (np.argmax(VGG_3B_16_64_l1l2_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_16_64_l1l2_0001 accuracy:  {VGG_3B_16_64_l1l2_0001_accuracy * 100:.2f} %")
    del VGG_3B_16_64_l1l2_0001


    VGG_3B_32_128_l1l2_0001.load_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1l2_0001").expect_partial()
    VGG_3B_32_128_l1l2_0001_pred = VGG_3B_32_128_l1l2_0001.predict(x_test)
    VGG_3B_32_128_l1l2_0001_accuracy = (np.argmax(VGG_3B_32_128_l1l2_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_32_128_l1l2_0001 accuracy: {VGG_3B_32_128_l1l2_0001_accuracy * 100:.2f} %")
    del VGG_3B_32_128_l1l2_0001




    FC_SP_16_256_l1l2_001.load_weights("./models/FC_SP_16_256/FC_SP_16_256_l1l2_001").expect_partial()
    FC_SP_16_256_l1l2_001_pred = FC_SP_16_256_l1l2_001.predict(x_test)
    FC_SP_16_256_l1l2_001_accuracy = (np.argmax(FC_SP_16_256_l1l2_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_SP_16_256_l1l2_001 accuracy:  {FC_SP_16_256_l1l2_001_accuracy * 100:.2f} %")
    del FC_SP_16_256_l1l2_001


    FC_MP_16_256_l1l2_001.load_weights("./models/FC_MP_16_256/FC_MP_16_256_l1l2_001").expect_partial()
    FC_MP_16_256_l1l2_001_pred = FC_MP_16_256_l1l2_001.predict(x_test)
    FC_MP_16_256_l1l2_001_accuracy = (np.argmax(FC_MP_16_256_l1l2_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_16_256_l1l2_001 accuracy:  {FC_MP_16_256_l1l2_001_accuracy * 100:.2f} %")
    del FC_MP_16_256_l1l2_001


    FC_MP_32_512_l1l2_001.load_weights("./models/FC_MP_32_512/FC_MP_32_512_l1l2_001").expect_partial()
    FC_MP_32_512_l1l2_001_pred = FC_MP_32_512_l1l2_001.predict(x_test)
    FC_MP_32_512_l1l2_001_accuracy = (np.argmax(FC_MP_32_512_l1l2_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_32_512_l1l2_001 accuracy:  {FC_MP_32_512_l1l2_001_accuracy * 100:.2f} %")
    del FC_MP_32_512_l1l2_001


    VGG_2B_32_64_l1l2_001.load_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l1l2_001").expect_partial()
    VGG_2B_32_64_l1l2_001_pred = VGG_2B_32_64_l1l2_001.predict(x_test)
    VGG_2B_32_64_l1l2_001_accuracy = (np.argmax(VGG_2B_32_64_l1l2_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_2B_32_64_l1l2_001 accuracy:  {VGG_2B_32_64_l1l2_001_accuracy * 100:.2f} %")
    del VGG_2B_32_64_l1l2_001


    VGG_3B_16_64_l1l2_001.load_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l1l2_001").expect_partial()
    VGG_3B_16_64_l1l2_001_pred = VGG_3B_16_64_l1l2_001.predict(x_test)
    VGG_3B_16_64_l1l2_001_accuracy = (np.argmax(VGG_3B_16_64_l1l2_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_16_64_l1l2_001 accuracy:  {VGG_3B_16_64_l1l2_001_accuracy * 100:.2f} %")
    del VGG_3B_16_64_l1l2_001


    VGG_3B_32_128_l1l2_001.load_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l1l2_001").expect_partial()
    VGG_3B_32_128_l1l2_001_pred = VGG_3B_32_128_l1l2_001.predict(x_test)
    VGG_3B_32_128_l1l2_001_accuracy = (np.argmax(VGG_3B_32_128_l1l2_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_32_128_l1l2_001 accuracy: {VGG_3B_32_128_l1l2_001_accuracy * 100:.2f} %")
    del VGG_3B_32_128_l1l2_001




    FC_SP_16_256_l2_00001.load_weights("./models/FC_SP_16_256/FC_SP_16_256_l2_00001").expect_partial()
    FC_SP_16_256_l2_00001_pred = FC_SP_16_256_l2_00001.predict(x_test)
    FC_SP_16_256_l2_00001_accuracy = (np.argmax(FC_SP_16_256_l2_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_SP_16_256_l2_00001 accuracy:  {FC_SP_16_256_l2_00001_accuracy * 100:.2f} %")
    del FC_SP_16_256_l2_00001


    FC_MP_16_256_l2_00001.load_weights("./models/FC_MP_16_256/FC_MP_16_256_l2_00001").expect_partial()
    FC_MP_16_256_l2_00001_pred = FC_MP_16_256_l2_00001.predict(x_test)
    FC_MP_16_256_l2_00001_accuracy = (np.argmax(FC_MP_16_256_l2_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_16_256_l2_00001 accuracy:  {FC_MP_16_256_l2_00001_accuracy * 100:.2f} %")
    del FC_MP_16_256_l2_00001


    FC_MP_32_512_l2_00001.load_weights("./models/FC_MP_32_512/FC_MP_32_512_l2_00001").expect_partial()
    FC_MP_32_512_l2_00001_pred = FC_MP_32_512_l2_00001.predict(x_test)
    FC_MP_32_512_l2_00001_accuracy = (np.argmax(FC_MP_32_512_l2_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_32_512_l2_00001 accuracy:  {FC_MP_32_512_l2_00001_accuracy * 100:.2f} %")
    del FC_MP_32_512_l2_00001


    VGG_2B_32_64_l2_00001.load_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l2_00001").expect_partial()
    VGG_2B_32_64_l2_00001_pred = VGG_2B_32_64_l2_00001.predict(x_test)
    VGG_2B_32_64_l2_00001_accuracy = (np.argmax(VGG_2B_32_64_l2_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_2B_32_64_l2_00001 accuracy:  {VGG_2B_32_64_l2_00001_accuracy * 100:.2f} %")
    del VGG_2B_32_64_l2_00001


    VGG_3B_16_64_l2_00001.load_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l2_00001").expect_partial()
    VGG_3B_16_64_l2_00001_pred = VGG_3B_16_64_l2_00001.predict(x_test)
    VGG_3B_16_64_l2_00001_accuracy = (np.argmax(VGG_3B_16_64_l2_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_16_64_l2_00001 accuracy:  {VGG_3B_16_64_l2_00001_accuracy * 100:.2f} %")
    del VGG_3B_16_64_l2_00001


    VGG_3B_32_128_l2_00001.load_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l2_00001").expect_partial()
    VGG_3B_32_128_l2_00001_pred = VGG_3B_32_128_l2_00001.predict(x_test)
    VGG_3B_32_128_l2_00001_accuracy = (np.argmax(VGG_3B_32_128_l2_00001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_32_128_l2_00001 accuracy: {VGG_3B_32_128_l2_00001_accuracy * 100:.2f} %")
    del VGG_3B_32_128_l2_00001




    FC_SP_16_256_l2_0001.load_weights("./models/FC_SP_16_256/FC_SP_16_256_l2_0001").expect_partial()
    FC_SP_16_256_l2_0001_pred = FC_SP_16_256_l2_0001.predict(x_test)
    FC_SP_16_256_l2_0001_accuracy = (np.argmax(FC_SP_16_256_l2_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_SP_16_256_l2_0001 accuracy:  {FC_SP_16_256_l2_0001_accuracy * 100:.2f} %")
    del FC_SP_16_256_l2_0001


    FC_MP_16_256_l2_0001.load_weights("./models/FC_MP_16_256/FC_MP_16_256_l2_0001").expect_partial()
    FC_MP_16_256_l2_0001_pred = FC_MP_16_256_l2_0001.predict(x_test)
    FC_MP_16_256_l2_0001_accuracy = (np.argmax(FC_MP_16_256_l2_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_16_256_l2_0001 accuracy:  {FC_MP_16_256_l2_0001_accuracy * 100:.2f} %")
    del FC_MP_16_256_l2_0001


    FC_MP_32_512_l2_0001.load_weights("./models/FC_MP_32_512/FC_MP_32_512_l2_0001").expect_partial()
    FC_MP_32_512_l2_0001_pred = FC_MP_32_512_l2_0001.predict(x_test)
    FC_MP_32_512_l2_0001_accuracy = (np.argmax(FC_MP_32_512_l2_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_32_512_l2_0001 accuracy:  {FC_MP_32_512_l2_0001_accuracy * 100:.2f} %")
    del FC_MP_32_512_l2_0001


    VGG_2B_32_64_l2_0001.load_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l2_0001").expect_partial()
    VGG_2B_32_64_l2_0001_pred = VGG_2B_32_64_l2_0001.predict(x_test)
    VGG_2B_32_64_l2_0001_accuracy = (np.argmax(VGG_2B_32_64_l2_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_2B_32_64_l2_0001 accuracy:  {VGG_2B_32_64_l2_0001_accuracy * 100:.2f} %")
    del VGG_2B_32_64_l2_0001


    VGG_3B_16_64_l2_0001.load_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l2_0001").expect_partial()
    VGG_3B_16_64_l2_0001_pred = VGG_3B_16_64_l2_0001.predict(x_test)
    VGG_3B_16_64_l2_0001_accuracy = (np.argmax(VGG_3B_16_64_l2_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_16_64_l2_0001 accuracy:  {VGG_3B_16_64_l2_0001_accuracy * 100:.2f} %")
    del VGG_3B_16_64_l2_0001


    VGG_3B_32_128_l2_0001.load_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l2_0001").expect_partial()
    VGG_3B_32_128_l2_0001_pred = VGG_3B_32_128_l2_0001.predict(x_test)
    VGG_3B_32_128_l2_0001_accuracy = (np.argmax(VGG_3B_32_128_l2_0001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_32_128_l2_0001 accuracy: {VGG_3B_32_128_l2_0001_accuracy * 100:.2f} %")
    del VGG_3B_32_128_l2_0001




    FC_SP_16_256_l2_001.load_weights("./models/FC_SP_16_256/FC_SP_16_256_l2_001").expect_partial()
    FC_SP_16_256_l2_001_pred = FC_SP_16_256_l2_001.predict(x_test)
    FC_SP_16_256_l2_001_accuracy = (np.argmax(FC_SP_16_256_l2_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_SP_16_256_l2_001 accuracy:  {FC_SP_16_256_l2_001_accuracy * 100:.2f} %")
    del FC_SP_16_256_l2_001


    FC_MP_16_256_l2_001.load_weights("./models/FC_MP_16_256/FC_MP_16_256_l2_001").expect_partial()
    FC_MP_16_256_l2_001_pred = FC_MP_16_256_l2_001.predict(x_test)
    FC_MP_16_256_l2_001_accuracy = (np.argmax(FC_MP_16_256_l2_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_16_256_l2_001 accuracy:  {FC_MP_16_256_l2_001_accuracy * 100:.2f} %")
    del FC_MP_16_256_l2_001


    FC_MP_32_512_l2_001.load_weights("./models/FC_MP_32_512/FC_MP_32_512_l2_001").expect_partial()
    FC_MP_32_512_l2_001_pred = FC_MP_32_512_l2_001.predict(x_test)
    FC_MP_32_512_l2_001_accuracy = (np.argmax(FC_MP_32_512_l2_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"FC_MP_32_512_l2_001 accuracy:  {FC_MP_32_512_l2_001_accuracy * 100:.2f} %")
    del FC_MP_32_512_l2_001


    VGG_2B_32_64_l2_001.load_weights("./models/VGG_2B_32_64/VGG_2B_32_64_l2_001").expect_partial()
    VGG_2B_32_64_l2_001_pred = VGG_2B_32_64_l2_001.predict(x_test)
    VGG_2B_32_64_l2_001_accuracy = (np.argmax(VGG_2B_32_64_l2_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_2B_32_64_l2_001 accuracy:  {VGG_2B_32_64_l2_001_accuracy * 100:.2f} %")
    del VGG_2B_32_64_l2_001


    VGG_3B_16_64_l2_001.load_weights("./models/VGG_3B_16_64/VGG_3B_16_64_l2_001").expect_partial()
    VGG_3B_16_64_l2_001_pred = VGG_3B_16_64_l2_001.predict(x_test)
    VGG_3B_16_64_l2_001_accuracy = (np.argmax(VGG_3B_16_64_l2_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_16_64_l2_001 accuracy:  {VGG_3B_16_64_l2_001_accuracy * 100:.2f} %")
    del VGG_3B_16_64_l2_001


    VGG_3B_32_128_l2_001.load_weights("./models/VGG_3B_32_128/VGG_3B_32_128_l2_001").expect_partial()
    VGG_3B_32_128_l2_001_pred = VGG_3B_32_128_l2_001.predict(x_test)
    VGG_3B_32_128_l2_001_accuracy = (np.argmax(VGG_3B_32_128_l2_001_pred, axis=1) == y_test).sum() / NUM_OF_TEST_SAMPLES
    print(f"VGG_3B_32_128_l2_001 accuracy: {VGG_3B_32_128_l2_001_accuracy * 100:.2f} %")
    del VGG_3B_32_128_l2_001
