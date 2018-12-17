def early_stop(train_loader, eval_loader, model, optimizer, criterion,
               maxepochs, device, n=1, patience=3):

    epoch_i = 0
    patience_i = 0
    best_validation_loss = float('inf')
    stop = 0

    # while the validation loss has not consistently increased
    while patience_i < patience:

        # train the model for n steps
        for i in range(n):

            if epoch_i == maxepochs:
                return stop

            train(epoch_i, train_loader, model, optimizer, criterion, device)
            epoch_i += 1

        # get the validation loss
        validation_loss = validate(epoch_i, eval_loader, model, criterion, device)

        if validation_loss < best_validation_loss:

            patience_i = 0
            best_validation_loss = validation_loss

            # save the model and update the stopping epoch
            checkpoint(args.outputfolder, epoch_i, model, best_validation_loss)
            stop = epoch_i

        else:
            patience_i += 1

    return stop
