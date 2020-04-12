if __name__=="__main__":
    model = Model()
    for epoch in range(50001):
        model.train(epoch)

    '''
    model.load('encoder.pth','decoder.pth')
    model.conditional_generation(0)
    #'''
