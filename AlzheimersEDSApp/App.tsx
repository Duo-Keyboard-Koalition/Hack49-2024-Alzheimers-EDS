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
        6;
      } catch (err) {
        console.warn(err);
      }
    }
  };

  // Start recording
  const startRecording = async (): Promise<void> => {
    try {
      setRecording(true);
      const path = `${RNFS.DocumentDirectoryPath}/${Date.now()}.wav`; // Use app's private storage
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

  // Function to upload audio to the server
  const uploadAudioToServer = async (uri: string) => {
    // Check if the file exists
    const fileExists = await RNFS.exists(uri);
    if (!fileExists) {
      console.error('File does not exist at the specified path:', uri);
      return;
    }

    const formData = new FormData();

    // Extract the file name from the recordingPath
    const fileName = uri.substring(uri.lastIndexOf('/') + 1);
    console.log('This is filename: ', fileName);

    formData.append('file', {
      uri: uri,
      type: 'audio/wav',
      name: fileName,
    });

    try {
      console.log('Uploading: ');
      // Phone : http://localhost:8801/uploadfile/
      // Emulator : http://10.0.2.2:8801/uploadfile/
      const response = await fetch('http://localhost:8801/uploadfile/', {
        method: 'POST',
        body: formData,
      });
      console.log('Success', response);
      const data = await response.json();
      console.log('Upload Response:', data); // Log the server response
    } catch (error) {
      console.error('Error uploading audio file:', error);
    }
  };

  // Classify the recording and upload to the server
  const classifyRecording = async (): Promise<void> => {
    if (!recordingPath) {
      console.warn('No recording available to classify.');
      return;
    }

    console.log('Classifying recording at:', recordingPath);

    // Call the upload function
    await uploadAudioToServer(recordingPath);
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
