def __generate_train_test_val_sets__(self):
    #generate train_test_validation files
    types_of_encoding = ["utf8", "cp1252"]
    fwh1 = open('./Ruber/RUBER/data_ttv/toronto_books_p2_query_train.txt','w')
    fwh2 = open('./Ruber/RUBER/data_ttv/toronto_books_p2_query_test.txt','w')
    fwh3 = open('./Ruber/RUBER/data_ttv/toronto_books_p2_query_valid.txt','w')
    fwg1 = open('./Ruber/RUBER/data_ttv/toronto_books_p2_reply_train.txt','w')
    fwg2 = open('./Ruber/RUBER/data_ttv/toronto_books_p2_reply_test.txt','w')
    fwg3 = open('./Ruber/RUBER/data_ttv/toronto_books_p2_reply_valid.txt','w')    
    frh = open('./Ruber/RUBER/data_ttv/toronto_books_p2_query.txt','r')
    frg = open('./Ruber/RUBER/data_ttv/toronto_books_p2_reply.txt','r')
    
    X = frh.readlines()
    Y = frg.readlines()
    
    X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.3, random_state=1) 
    X_train, X_val, Y_train, Y_val    = train_test_split(X_train, Y_train, test_size=0.125, random_state=1)     
    print(len(X_train))
    print(len(Y_train))
    print(len(X_test))
    print(len(Y_test))        
    print(len(X_val))
    print(len(Y_val)) 
    fwh1.write("".join(X_train))
    fwh2.write("".join(X_test))
    fwh3.write("".join(X_val))
    fwg1.write("".join(Y_train))
    fwg2.write("".join(Y_test))
    fwg3.write("".join(Y_val))

__generate_train_test_val_sets__()