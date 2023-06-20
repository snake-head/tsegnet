<template>
  <div id='display'>
    <div ref="vtkContainer"/>
    <div id="controls" v-if = 'ifControls'>
      <el-checkbox v-for='count in 15' :key=count v-model=checkedDisplay[count-1] :label=labelList[count-1] />
    </div>
  </div>
</template>

<script>
  import { ref, onMounted, onBeforeUnmount, computed, watchEffect, watch} from 'vue';
  import '@kitware/vtk.js/Rendering/Profiles/Geometry';
  import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
  import vtkSTLReader from '@kitware/vtk.js/IO/Geometry/STLReader'

  import vtkActor           from '@kitware/vtk.js/Rendering/Core/Actor';
  import vtkMapper          from '@kitware/vtk.js/Rendering/Core/Mapper';
  import vtkSphereSource from '@kitware/vtk.js/Filters/Sources/SphereSource';
  import macro from '@kitware/vtk.js/macros';
  import $store from "../store/index"

    export default {
        name:'Display',
        setup(){
          let currentUrl = computed(()=>{
            return $store.state.currentUrl
          })
          let ifDisplayCentroids = computed(()=>{
            return $store.state.ifDisplayCentroids
          })
          let ifDisplayResult = computed(()=>{
            return $store.state.ifDisplayResult
          })
          const vtkContainer = ref(null);
          let checkedDisplay = ref([true,true,true,true,true,true,true,true,true,true,true,true,true,true,true])
          let labelList = ref(['显示牙龈','显示右1','显示右2','显示右3','显示右4','显示右5','显示右6','显示右7',
          '显示左1','显示左2','显示左3','显示左4','显示左5','显示左6','显示左7',])
          let ifControls = ref(false)
          let actorList = ref([])

          function cleanWindow(){
            if ($store.state.vtkContext) {
              const { genericRenderer, actor, mapper } = $store.state.vtkContext;
              actor.delete();
              mapper.delete();
              genericRenderer.delete();
              // vtkContext.value = null;
              $store.dispatch('setVtkContext', null)
            }
          }

          watch(currentUrl,()=>{
            // console.log('监视到变化')
            // console.log(currentUrl.value)
            if (!currentUrl.value){
              return
            }
            const reader = vtkSTLReader.newInstance();
            const mapper = vtkMapper.newInstance({ scalarVisibility: false });
            const actor = vtkActor.newInstance();

            actor.setMapper(mapper);
            mapper.setInputConnection(reader.getOutputPort());

            const genericRenderer = vtkGenericRenderWindow.newInstance({});
            genericRenderer.setContainer(vtkContainer.value)
            genericRenderer.resize()

            const renderer = genericRenderer.getRenderer();

            const renderWindow = genericRenderer.getRenderWindow();
            const resetCamera = renderer.resetCamera;
            const render = renderWindow.render;
            
            reader.setUrl(currentUrl.value, { binary: true }).then(
              function update() {
                
                // console.log(vtkContainer.value)
                renderer.addActor(actor);

                resetCamera();
                render();
                
                $store.dispatch('setVtkContext', {
                  genericRenderer,
                  renderWindow,
                  renderer,
                  // coneSource,
                  actor,
                  mapper,
                })
              }
            );
          })

          watch(ifDisplayCentroids,()=>{
            if (ifDisplayCentroids){
              const centroidsPred = $store.state.centroidsPred
              const { genericRenderer } = $store.state.vtkContext;
              const renderWindow = genericRenderer.getRenderWindow();
              const renderer = genericRenderer.getRenderer();
              const resetCamera = renderer.resetCamera;
              const render = renderWindow.render;

              // 只显示一组（14个质心）
              for(var i=0;i<14;i++){
                const Centroid=centroidsPred[i];
                const sphereSource = vtkSphereSource.newInstance();
                sphereSource.setCenter(Centroid)
                sphereSource.setRadius(1)
                const actor = vtkActor.newInstance();
                actor.getProperty().setColor(0, 0, 1)
                const mapper = vtkMapper.newInstance();
                mapper.setInputConnection(sphereSource.getOutputPort());
                actor.setMapper(mapper);
                renderer.addActor(actor);
              }
              resetCamera();
              render();
            }
          })

          watch(ifDisplayResult,()=>{
            if (ifDisplayResult){
              // 清空原窗口
              // console.log('clear')
              cleanWindow()
              ifControls.value = true //显示控制面板
              //渲染
              let filename = currentUrl.value.split('/').slice(-1)[0].split('.')[0]
              console.log(filename)
              let resultPath = `http://127.0.0.1:8000/tooth/upperResult/${filename}`
              console.log(resultPath)
              const genericRenderer = vtkGenericRenderWindow.newInstance({});
              genericRenderer.setContainer(vtkContainer.value)
              genericRenderer.resize()

              const renderer = genericRenderer.getRenderer();

              const renderWindow = genericRenderer.getRenderWindow();
              const resetCamera = renderer.resetCamera;
              const render = renderWindow.render;

              //递归，串行地生成每个单齿的actor
              function requestStl(i=0){
                if (i>14){
                  return
                }else{
                  const reader = vtkSTLReader.newInstance();
                  reader.setUrl(resultPath+'/'+i+'.stl', { binary: true }).then(
                    ()=> {
                      const mapper = vtkMapper.newInstance({ scalarVisibility: false });
                      const actor = vtkActor.newInstance();
                      actor.getProperty().setColor(Math.random(), Math.random(), Math.random())
                      actor.setMapper(mapper);
                      mapper.setInputConnection(reader.getOutputPort());

                      // console.log(vtkContainer.value)
                      renderer.addActor(actor);

                      resetCamera();
                      render();
                      
                      
                      actorList.value.push(actor)

                      $store.dispatch('setVtkContext', {
                      genericRenderer,
                      renderWindow,
                      renderer,
                      // coneSource,
                      actor,
                      mapper,
                      })
                      requestStl(i+1)
                    }
                  )
                }
              }
              requestStl()
            }
          },{deep:true})

          watch(checkedDisplay,()=>{
            for(var i=0;i<15;i++){
              var opacity = checkedDisplay.value[i] ? 1 : 0
              let actor = actorList.value[i]
              actor.getProperty().setOpacity(opacity)
              const renderWindow = $store.state.vtkContext.renderWindow
              renderWindow.render()
            }
          },{
            deep:true,
            })

          onBeforeUnmount(() => {
            if ($store.state.vtkContext) {
              const { genericRenderer, actor, mapper } = $store.state.vtkContext;
              actor.delete();
              mapper.delete();
              genericRenderer.delete();
              // vtkContext.value = null;
              $store.dispatch('setVtkContext', null)
            }
          });

          return {
            vtkContainer,
            cleanWindow,
            checkedDisplay,
            ifControls,
            labelList,
            actorList,
          };
      }
    }
</script>
<style >
  #display{
    display: block;
  }
  #controls {
    position: absolute;
    top: 40px;
    left: 440px;
    background: rgb(82,87,110);
    padding: 12px;
  }
  .el-checkbox{
    display: block;
    color: aliceblue;
  }
</style>