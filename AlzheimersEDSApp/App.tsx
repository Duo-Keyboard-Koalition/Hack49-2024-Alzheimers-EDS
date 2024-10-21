import React, {useEffect, useState} from 'react';
import {
  View,
  Button,
  Text,
  StyleSheet,
  Platform,
  PermissionsAndroid,
} from 'react-native';
import Sound from 'react-native-sound';
import AudioRecorderPlayer from 'react-native-audio-recorder-player';
import RNFS from 'react-native-fs'; // Import react-native-fs
import RNFetchBlob from 'rn-fetch-blob';
import notifee, { AndroidImportance } from '@notifee/react-native'; // Import Notifee

const audioRecorderPlayer = new AudioRecorderPlayer();

function App(): React.JSX.Element {
  const [recording, setRecording] = useState<boolean>(false);
  const [recordingPath, setRecordingPath] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [audio, setAudio] = useState<Sound | null>(null); // Update the state type to allow Sound or null

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

        // Request notification permission
        const notificationPermission = await notifee.requestPermission();
        if (!notificationPermission) {
          console.warn('Notification permission denied');
          return; // Exit if notification permission is denied
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
      const path = `${RNFS.DocumentDirectoryPath}/${Date.now()}.m4a`; // Use app's private storage
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
    // console.log('This is filename: ', fileName);

    formData.append('file', {
      uri: uri,
      type: 'audio/wav',
      name: fileName,
    });

    try {
      console.log('Uploading: ');
      // Phone : http://localhost:8801/uploadfile/
      // Emulator : http://10.0.2.2:8801/uploadfile/
      const response = await fetch('http://localhost:8000/uploadfile/', {
        method: 'POST',
        body: formData,
      });
      console.log('Success', response);
      const data = await response.json();

      fetchAudio();
      // console.log('Upload Response:', data); // Log the server response
      return data;
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

    const channelId = await notifee.createChannel({
      id: 'default',
      name: 'Default Channel',
    });

    const result = await uploadAudioToServer(recordingPath);
    console.log("Model output: " + result['prediction'])

    if (result['prediction'] >= 0.5) {
      await notifee.displayNotification({
        title: "Health Checkup",
        // body: `The classification prediction is: ${result['prediction'] >= 0.5 ? 'Positive' : 'Negative'}`,
        body: "Hi - we noticed you've been showing subtle signs of mental decline recently and we think it's time for a checkup.",
        android: {
          channelId: channelId,
          importance: AndroidImportance.HIGH,
        },
      });
    }
  };

  // const fetchAudio = async () => {
  //   console.log('fetching audio');
  //   const audioUrl = 'http://localhost:8000/get-audio';

  //   RNFetchBlob.config({
  //     fileCache: true,
  //     path: RNFetchBlob.fs.dirs.DocumentDir + '/output.mp3', // Path to save the file
  //   })
  //     .fetch('GET', audioUrl)
  //     .then((res) => {
  //       console.log('playing' + res.path())
  //       // Step 2: Play the audio file
  //       const sound = new Sound('http://localhost:8000/get-audio', Sound.MAIN_BUNDLE, (error) => {
  //         console.log('sound')
  //         if (error) {
  //           console.error('Failed to load sound', error);
  //           return;
  //         }
  //         // Play the sound
  //         sound.play((success) => {
  //           if (!success) {
  //             console.log('Sound did not play');
  //           }
  //         });
  //       });
  //       console.log(sound)

  //       console.log('done')
  //     })
  //     .catch((error) => {
  //       console.error('Error fetching audio', error);
  //     });
  // };

  const fetchAudio = async () => {
    const audioUrl = 'http://localhost:8000/get-audio'; // Update this URL if needed
    console.log('Playing audio from:', audioUrl);

    const sound = new Sound(audioUrl, Sound.MAIN_BUNDLE, (error) => {
      if (error) {
        console.error('Failed to load sound', error);
        return;
      }
      console.log('Sound loaded successfully:', sound);
      
      // Play the sound
      sound.play((success) => {
        if (!success) {
          console.log('Sound did not play');
        }
      });
    });
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
