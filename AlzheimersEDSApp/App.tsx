import React, {useEffect, useState} from 'react';
import {
  View,
  Button,
  Text,
  StyleSheet,
  Platform,
  PermissionsAndroid,
} from 'react-native';
import AudioRecorderPlayer from 'react-native-audio-recorder-player';
import RNFS from 'react-native-fs'; // Import react-native-fs

const audioRecorderPlayer = new AudioRecorderPlayer();

function App(): React.JSX.Element {
  const [recording, setRecording] = useState<boolean>(false);
  const [recordingPath, setRecordingPath] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  // Request Android permissions
  const requestPermissions = async (): Promise<void> => {
    if (Platform.OS === 'android') {
      try {
        const audioPermission = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
        );
        if (audioPermission !== PermissionsAndroid.RESULTS.GRANTED) {
          console.warn('Audio permission denied');
          return; // Exit if audio permission is denied
        }
      } catch (err) {
        console.warn(err);
      }
    }
  };

  // Start recording
  const startRecording = async (): Promise<void> => {
    try {
      setRecording(true);
      const path = `${RNFS.DocumentDirectoryPath}/${Date.now()}.mp3`; // Use app's private storage
      const result = await audioRecorderPlayer.startRecorder(path);
      setRecordingPath(result);
      audioRecorderPlayer.addRecordBackListener(e => {
        console.log('Recording...', e.currentPosition);
      });
    } catch (error) {
      console.warn('Error starting recording:', error);
    }
  };

  // Stop recording
  const stopRecording = async (): Promise<void> => {
    try {
      const result = await audioRecorderPlayer.stopRecorder();
      setRecording(false);
      audioRecorderPlayer.removeRecordBackListener();
      console.log('Recording stopped:', result);
    } catch (error) {
      console.warn('Error stopping recording:', error);
    }
  };

  // Play recorded audio
  const playRecording = async (): Promise<void> => {
    if (!recordingPath) return;

    try {
      setIsPlaying(true);
      await audioRecorderPlayer.startPlayer(recordingPath);
      audioRecorderPlayer.addPlayBackListener(e => {
        if (e.currentPosition === e.duration) {
          setIsPlaying(false);
          audioRecorderPlayer.stopPlayer();
        }
      });
    } catch (error) {
      console.warn('Error playing recording:', error);
    }
  };

  // Stop playing audio
  const stopPlaying = async (): Promise<void> => {
    try {
      await audioRecorderPlayer.stopPlayer();
      setIsPlaying(false);
      audioRecorderPlayer.removePlayBackListener();
    } catch (error) {
      console.warn('Error stopping playback:', error);
    }
  };

  // Classify the recording (placeholder function)
  const classifyRecording = async (): Promise<void> => {
    if (!recordingPath) {
      console.warn('No recording available to classify.');
      return;
    }
    console.log('Classifying recording at:', recordingPath);
    // Implement your classification logic here
  };

  useEffect(() => {
    requestPermissions();
  }, []);

  return (
    <View style={styles.container}>
      <Text style={styles.appName}>RecallAI</Text>
      <Button
        title={recording ? 'Stop Recording' : 'Start Recording'}
        onPress={recording ? stopRecording : startRecording}
      />
      {recording ? (
        <Text style={styles.recordingText}>Recording...</Text>
      ) : null}
      {recordingPath ? (
        <>
          <Text style={styles.pathText}>
            Recording saved to: {recordingPath}
          </Text>
          <Button
            title={isPlaying ? 'Stop Playback' : 'Play Recording'}
            onPress={isPlaying ? stopPlaying : playRecording}
            disabled={recording} // Disable when recording
          />
          <View style={styles.classifyButtonContainer}>
            <Button
              title="Classify"
              onPress={classifyRecording}
              disabled={recording} // Disable when recording
            />
          </View>
        </>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  appName: {
    fontSize: 30,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  recordingText: {
    fontSize: 20,
    color: 'red',
    marginVertical: 20,
  },
  pathText: {
    marginVertical: 20,
  },
  classifyButtonContainer: {
    marginTop: 20,
  },
});

export default App;
