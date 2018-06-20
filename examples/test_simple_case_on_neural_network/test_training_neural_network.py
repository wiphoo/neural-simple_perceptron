#!/usr/bin/env python

import numpy

import Perceptron
import ForwardPropagator
import BackwardPropagator

#	globals
#Tables_AND = ( #	inputs
#				numpy.array( [ [ 0, 0 ], 
#								[ 0, 1 ],
#								[ 1, 0 ],
#								[ 1, 1 ], ] ),
#				#	outputs
#				numpy.array( [ [ 0, ],
#								[ 0, ],
#								[ 0, ],
#								[ 1 ], ] ) )
Tables = ( #	inputs
				numpy.array( [ [ 0, 0, 1, 1 ], 
								[ 0, 1, 1, 1 ],
								[ 1, 0, 1, 1 ],
								[ 1, 1, 1, 1 ], ] ),
				#	outputs
				numpy.array( [ [ 0, ],
								[ 1, ],
								[ 1, ],
								[ 0, ], ] ) )


InputsMatrix = Tables[0]
OutputsMatrix = Tables[1]
NumHiddenLayersNeurons = 2
LearningRate = 0.5   

NumTraining = 1
NumPrintRMSError = int( NumTraining / 25. )

NumInputs = InputsMatrix.shape[1] - 1
NumOuputs = OutputsMatrix.shape[1]
NumSamples = OutputsMatrix.shape[0]
assert( InputsMatrix.shape[0] == OutputsMatrix.shape[0] )

#	main
def main():
	print( 'main()...........' )
	print( '           NumInputs = {}'.format( NumInputs ) )
	print( '           NumHiddenLayersNeurons = {}'.format( NumHiddenLayersNeurons ) )
	print( '           NumOuputs = {}'.format( NumOuputs ) )
	print( '           NumSamples = {}'.format( NumSamples ) )
	
	print( '           InputsMatrix = {}'.format( InputsMatrix ) )
	print( '           OutputsMatrix = {}'.format( OutputsMatrix ) )
	
	

	
	#	create perceptron
	perceptron = Perceptron.Perceptron( NumInputs, NumHiddenLayersNeurons, NumOuputs, NumSamples )
	print( 'PERCEPTRON' )
	print( '   hidden layer neurons' )
	print( '            perceptron.hiddenLayerWeightsMatrix = {}'.format( perceptron.hiddenLayerWeightsMatrix ) )
	print( '            perceptron.hiddenLayerBiasMatrix = {}'.format( perceptron.hiddenLayerBiasMatrix ) )
	print( '   outputs' )
	print( '            perceptron.outputLayerWeightsMatrix = {}'.format( perceptron.outputLayerWeightsMatrix ) )
	print( '            perceptron.outputLayerBiasMatrix = {}'.format( perceptron.outputLayerBiasMatrix ) )

	print( 'LOOP TRAINING...................................' )
	for i in range( NumTraining ):
	
		#	create forward propagate
		forwardPropagator = ForwardPropagator.ForwardPropagator( InputsMatrix )

		#	do forward propagate
		forwardPropagator.propagate( perceptron )
		print( 'FORWARD PROPAGATE' )
		print( '            forwardPropagate.inputsMatrix = {}'.format( forwardPropagator.inputsMatrix ) )
		print( '            forwardPropagate.outputsMatrixHiddenLayerNeurons = {}'.format( forwardPropagator.outputsMatrixHiddenLayerNeurons ) )
		print( '            forwardPropagate.outputsMatrixOutputsLayer = {}'.format( forwardPropagator.outputsMatrixOutputsLayer ) )


#		#	create and do a back propagate
#		backwardPropagator = BackwardPropagator.BackwardPropagator()
#		backwardPropagator.propagate( perceptron, forwardPropagator, OutputsMatrix, LearningRate )
#
##		print( 'BACKWARD PROPAGATE' )
##		print( '            backwardPropagator.hiddenLayerDeltaWeightsMatrix = {}'.format( 
##												backwardPropagator.hiddenLayerDeltaWeightsMatrix ) )
##		print( '            backwardPropagator.outputDeltaWeightsMatrix = {}'.format( 
##												backwardPropagator.outputDeltaWeightsMatrix ) )
#
#		#	apply delta weights
#		perceptron.applyDeltaWeights( backwardPropagator )
#		
#		if i % NumPrintRMSError == 0:
#			print( '####    iter[{}]'.format( i ) )
#			print( '####    error = {}'.format( backwardPropagator.calculateRmsError() ) )
		
		
		
	
if __name__ == '__main__':
	main()
